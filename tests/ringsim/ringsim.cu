// ringsim: a simulator of the prompt's grouped MoE block with its experts streamed through a VRAM ring
// (moe-cache.cu ggml_cuda_moe_ring_block, mmq.cu moe_block_grouped_impl), for exploring its pipeline on a real GPU.
//
// What is real: the copies (cudaMemcpyAsync host -> device from registered host memory, each expert's stride, into the
// entries of a ring of the given size, on a copy stream), the streams, the events and their waits, the host threads
// and their launches (8 per group as in the grouped block, the dense part in K launches), the per-block readback and
// synchronization. What is modeled: the kernels' work, as blocks that each spin for a calibrated time (MMQ: one block
// per 128-row tile of output x 128 tokens of an expert, J = 128, the occupancy forced by the same shared memory as
// MMQ), so the grid of each group and its wave quantization are the hardware's. The routing (rows of each expert,
// which experts the cache held) comes from a dump of the real block (GGML_CUDA_MOE_RING_DUMP=FILE) or is synthetic.
// The default kernel times are those of an RTX 4090 trace (19k prompt, -ub 9536, 2026-11-08).
//
// The ring is the real one's: entries taken in turn in (layer, expert) order for the experts the cache does not hold,
// a copy waits (device side) for the event of the group that read its entry last; a group waits for the event after
// its last copy. Each copy's data carries a tag at both ends, checked by the group's first kernel (arrived) and after
// its down product (not overwritten): errors must be 0.
//
// Issuers of the copies:
//   block   the real one: the thread that evaluates the graph, at its block, the block's copies whose wait is recorded,
//           then more after each group (the copies of block L+1 can only be queued once the graph thread reaches it)
//   ahead   the graph thread, but over the whole queue: every copy whose wait is recorded, future blocks included
//   thread  a dedicated thread walking the whole queue, waiting (host side) for each wait to be recorded
//   hthread a dedicated thread that waits for the read to be done (cudaEventSynchronize), no device-side wait
// --sync 1: the block reads its bounds back and synchronizes the compute stream before its groups (the real code);
// --sync 0: as if the bounds were known on the host without waiting (the bound of a device-side grid).
//
// Build (CUDA):  nvcc -O2 -std=c++17 -arch=sm_89 -o ringsim tests/ringsim/ringsim.cu -lpthread
// Build (HIP):   hipcc -O2 -std=c++17 --offload-arch=gfx1100 -x hip -o ringsim tests/ringsim/ringsim.cu -lpthread
// Run:           ./ringsim --dump routing.bin --G 16,32 --ring 900,1800 --issuer block,thread --sync 1,0
//                ./ringsim --help

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamDestroy hipStreamDestroy
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventDestroy hipEventDestroy
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostUnregister hipHostUnregister
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaGetLastError hipGetLastError
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#else
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <set>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

// ---------------------------------------------------------------------------------------------------------------------
// device side

__device__ __forceinline__ uint64_t now_ns(uint64_t ticks_per_us) {
#if defined(__HIP_PLATFORM_AMD__)
    return wall_clock64() * 1000 / ticks_per_us;
#else
    (void) ticks_per_us;
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

// every block spins ns (its occupancy set by the dynamic shared memory of the launch)
__global__ void k_spin(uint64_t ns, uint64_t tpu) {
    extern __shared__ char sm[];
    if (threadIdx.x == 0) {
        const uint64_t t0 = now_ns(tpu);
        while (now_ns(tpu) - t0 < ns) {
        }
        sm[0] = 0;
    }
    __syncthreads();
}

// a group's tags: each absent expert's entry must hold its data at both ends (the copy arrived, nothing overwrote it)
__global__ void k_check(const uint64_t * addr, const uint64_t * tag, int n, uint64_t stride, int * errors, uint64_t ns, uint64_t tpu) {
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        const uint64_t * p = (const uint64_t *) addr[k];
        const uint64_t * q = (const uint64_t *) (addr[k] + stride - 8);
        if (*(volatile const uint64_t *) p != tag[k] || *(volatile const uint64_t *) q != tag[k]) {
            atomicAdd(errors, 1);
        }
    }
    if (threadIdx.x == 0 && ns) {
        const uint64_t t0 = now_ns(tpu);
        while (now_ns(tpu) - t0 < ns) {
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the routing

struct block_rec {
    int layer = 0, ntok = 0, nexp = 0;
    int pass = 0;   // the prompt pass (a pass's last layer may be missing: it computes the output rows only)
    uint64_t stride = 0;
    int tg = 0, td = 0;
    std::vector<int> rows;
    std::vector<uint8_t> held;
};

static std::vector<block_rec> load_dump(const char * path) {
    std::vector<block_rec> v;
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    for (;;) {
        int32_t hdr[4];
        if (fread(hdr, sizeof(hdr), 1, f) != 1) {
            break;
        }
        if (hdr[0] != 0x31474e52) {
            fprintf(stderr, "%s: bad record\n", path);
            exit(1);
        }
        block_rec b;
        b.layer = hdr[1]; b.ntok = hdr[2]; b.nexp = hdr[3];
        int32_t types[2];
        if (fread(&b.stride, 8, 1, f) != 1 || fread(types, sizeof(types), 1, f) != 1) {
            break;
        }
        b.tg = types[0]; b.td = types[1];
        b.rows.resize(b.nexp);
        b.held.resize(b.nexp);
        if (fread(b.rows.data(), 4, b.nexp, f) != (size_t) b.nexp || fread(b.held.data(), 1, b.nexp, f) != (size_t) b.nexp) {
            break;
        }
        v.push_back(std::move(b));
    }
    fclose(f);
    return v;
}

// k consecutive prompt passes of each layer as one (rows summed: a pass of k times the tokens routes the same way)
static std::vector<block_rec> merge_passes(const std::vector<block_rec> & in, int k) {
    if (k <= 1) {
        return in;
    }
    std::vector<std::vector<block_rec>> passes;
    for (const block_rec & b : in) {
        if (passes.empty() || b.layer <= passes.back().back().layer) {
            passes.push_back({});
        }
        passes.back().push_back(b);
    }
    std::vector<block_rec> out;
    for (size_t p = 0; p + k <= passes.size(); p += k) {
        for (size_t l = 0; l < passes[p].size(); l++) {
            block_rec m = passes[p][l];
            bool ok = true;
            for (int j = 1; j < k; j++) {
                if (passes[p + j].size() != passes[p].size() || passes[p + j][l].layer != m.layer) {
                    ok = false;
                    break;
                }
                m.ntok += passes[p + j][l].ntok;
                for (int e = 0; e < m.nexp; e++) {
                    m.rows[e] += passes[p + j][l].rows[e];
                }
            }
            if (ok) {
                out.push_back(std::move(m));
            }
        }
    }
    return out;
}

// a synthetic prompt: 48 layers, Zipf-like routing of ntok x 10 rows over 512 experts, the hottest share in the cache
static std::vector<block_rec> synthetic(int passes, int ntok, double held_share, unsigned seed) {
    std::vector<block_rec> v;
    srand(seed);
    for (int p = 0; p < passes; p++) {
        for (int l = 0; l < 48; l++) {
            block_rec b;
            b.layer = l; b.ntok = ntok; b.nexp = 512;
            const bool k3 = l < 4 || l >= 40 || (l % 3 == 0);
            b.stride = k3 ? 2253056 : 1971200;
            b.tg = k3 ? 44 : 22; b.td = 20;
            std::vector<double> w(512);
            std::vector<int> perm(512);
            for (int e = 0; e < 512; e++) {
                perm[e] = e;
            }
            for (int e = 511; e > 0; e--) {
                std::swap(perm[e], perm[rand() % (e + 1)]);
            }
            double s = 0;
            std::vector<int> rank(512);
            for (int r = 0; r < 512; r++) {
                w[perm[r]] = 1.0 / pow(r + 8.0, 0.8);   // expert perm[r] has rank r
                rank[perm[r]] = r;
                s += w[perm[r]];
            }
            b.rows.resize(512);
            b.held.assign(512, 0);
            for (int e = 0; e < 512; e++) {
                b.rows[e] = (int) (w[e] / s * ntok * 10.0 + 0.5);
                b.held[e] = rank[e] < held_share * 512 ? 1 : 0;   // the hottest share
            }
            v.push_back(std::move(b));
        }
    }
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// the timing model (us per MMQ block of 128 x 128, RTX 4090 trace of 2026-11-08, -ub 9536)

struct calib {
    std::map<int, double> t_gu = { {22, 59.7}, {44, 60.7}, {43, 80.7}, {21, 70.0}, {18, 65.0} };
    std::map<int, double> t_d  = { {20, 21.7}, {42, 21.4}, {23, 22.0} };
    double t_gu_default = 60.0, t_d_default = 21.7;
    int tiles_gu = 5, tiles_d = 20;     // n_ff / 128 (640), n_embd / 128 (2560)
    int J = 128;                        // MMQ's tokens per tile on NVIDIA (ncols_opt = n_tok)
    double dense_ns_tok = 3600.0;       // the rest of a layer: 34.3 ms at 9 536 tokens
    double route_ns_tok = 91.0;         // mm_ids_helper + the activation's quantization: 0.87 ms at 9 536
    double glu_ns_row = 13.1;           // swiglu + down's activation quantization, per routed row
    double prefix_us = 2.0;             // k_mmq_tile_prefix of a group
    int dense_kernels = 64;
    double kscale = 1.0;                // every kernel time times this
};

// ---------------------------------------------------------------------------------------------------------------------
// one run

struct config {
    int G = 32;
    int ring_mib = 900;
    std::string issuer = "block";
    int sync = 1;
    int wait1 = 0;          // 1: the copy stream waits once per group read (the waits only grow), not once per copy
    bool compute = true, copies = true;
};

struct copy_op {
    int64_t group;      // the group that reads it (global sequence)
    int64_t wait;       // the group whose read must end first (-1: none)
    int block, expert, entry;
};

struct plan {
    std::vector<copy_op> ops;                              // every copy, in order
    std::vector<size_t> block_first, block_end;            // each block's ops
    std::vector<std::vector<uint64_t>> addr, tag;          // per group: the entries of its copied experts and their tags
    std::vector<int64_t> first_op_of_group;                // per group: its first op (or -1), last op
    std::vector<int64_t> last_op_of_group;
    int64_t n_groups = 0;
    size_t entry = 0, n_entries = 0;
    uint64_t in_place = 0;
};

struct sim {
    std::vector<block_rec> blocks;
    calib cal;
    int n_sm = 0;
    uint64_t tpu = 1000;                    // HIP: wall clock ticks per us
    size_t smem_gu = 61952, smem_d = 57856, smem_dense = 0;
    char * src = nullptr;                   // registered host memory: n_src experts of max stride, tagged
    size_t src_stride = 0;
    int n_src = 0;
    char * ring = nullptr;
    size_t ring_bytes = 0;
    int * d_err = nullptr;
    uint64_t * d_tabs = nullptr;            // per group: addresses then tags (n_groups x 2 x G)
    size_t d_tabs_n = 0;
    cudaStream_t cs = nullptr, cp = nullptr;
    bool check = true;
    bool verbose = false;

    uint64_t ns(double us) const { return (uint64_t) (us * 1000.0 * cal.kscale); }

    static int64_t global_expert(const block_rec & b, int e) { return ((int64_t) b.pass * 256 + b.layer) * b.nexp + e; }

    uint64_t tag_of(int64_t global_expert) const { return 0x5a5a000000000000ull | (uint64_t) (global_expert % n_src); }

    plan make_plan(const config & c) const {
        plan p;
        size_t largest = 0;
        for (const block_rec & b : blocks) {
            largest = std::max<size_t>(largest, b.stride);
        }
        p.entry = (largest + 1024 + 255) / 256 * 256;
        p.n_entries = (size_t) c.ring_mib * 1024 * 1024 / p.entry;
        const int ng = blocks[0].nexp / c.G;
        p.n_groups = (int64_t) blocks.size() * ng;
        p.addr.resize(p.n_groups);
        p.tag.resize(p.n_groups);
        p.first_op_of_group.assign(p.n_groups, -1);
        p.last_op_of_group.assign(p.n_groups, -1);
        std::vector<int64_t> reader(p.n_entries, -1);
        size_t next = 0;
        for (size_t bi = 0; bi < blocks.size(); bi++) {
            const block_rec & b = blocks[bi];
            p.block_first.push_back(p.ops.size());
            for (int e = 0; e < b.nexp; e++) {
                if (b.held[e]) {
                    p.in_place++;
                    continue;
                }
                const int64_t g = (int64_t) bi * ng + e / c.G;
                copy_op op;
                op.group = g; op.block = (int) bi; op.expert = e; op.entry = (int) next;
                op.wait = reader[next];
                if (op.wait >= g) {
                    fprintf(stderr, "ring of %zu entries too small for groups of %d\n", p.n_entries, c.G);
                    exit(1);
                }
                reader[next] = g;
                next = (next + 1) % p.n_entries;
                if (p.first_op_of_group[g] < 0) {
                    p.first_op_of_group[g] = (int64_t) p.ops.size();
                }
                p.last_op_of_group[g] = (int64_t) p.ops.size();
                p.addr[g].push_back((uint64_t) (uintptr_t) (ring + (size_t) op.entry * p.entry));
                p.tag[g].push_back(tag_of(global_expert(b, e)));
                p.ops.push_back(op);
            }
            p.block_end.push_back(p.ops.size());
        }
        return p;
    }

    // the host source of an op: its expert's tagged data (n_src > the ring's entries + a layer: a stale entry differs)
    const char * src_of(const copy_op & op) const {
        const block_rec & b = blocks[op.block];
        return src + (size_t) (global_expert(b, op.expert) % n_src) * src_stride;
    }

    void launch_spin(int grid, size_t smem, double us) {
        if (grid <= 0 || us <= 0) {
            return;
        }
        k_spin<<<grid, 256, smem, cs>>>(ns(us), tpu);
    }

    // the routing part (mm_ids_helper, quantization) and the dense part before it (attention, GDN, ...)
    void launch_dense(const block_rec & b) {
        const double us = cal.dense_ns_tok * b.ntok / 1000.0 / cal.dense_kernels;
        for (int k = 0; k < cal.dense_kernels; k++) {
            launch_spin(n_sm, smem_dense, us);
        }
        launch_spin(n_sm, smem_dense, cal.route_ns_tok * b.ntok / 1000.0 / 2);
        launch_spin(n_sm, smem_dense, cal.route_ns_tok * b.ntok / 1000.0 / 2);
    }

    // a group's 8 launches (3 tile prefixes, gate, up, swiglu, quantization, down), its check after the down product
    void launch_group(const block_rec & b, int g0, int G, const plan & p, int64_t gseq) {
        int tiles = 0, rows = 0;
        for (int e = g0; e < g0 + G; e++) {
            tiles += (b.rows[e] + cal.J - 1) / cal.J;
            rows += b.rows[e];
        }
        if (rows == 0) {
            return;
        }
        const double tgu = cal.t_gu.count(b.tg) ? cal.t_gu.at(b.tg) : cal.t_gu_default;
        const double td = cal.t_d.count(b.td) ? cal.t_d.at(b.td) : cal.t_d_default;
        const int n = (int) p.addr[gseq].size();
        uint64_t * tab = d_tabs + (size_t) gseq * 2 * G;
        if (check && n > 0) {
            k_check<<<1, 64, 0, cs>>>(tab, tab + G, n, b.stride, d_err, ns(cal.prefix_us), tpu);
        } else {
            launch_spin(1, 0, cal.prefix_us);
        }
        launch_spin(1, 0, cal.prefix_us);
        launch_spin(1, 0, cal.prefix_us);
        launch_spin(tiles * cal.tiles_gu, smem_gu, tgu);
        launch_spin(tiles * cal.tiles_gu, smem_gu, tgu);
        launch_spin(n_sm, smem_dense, cal.glu_ns_row * rows / 1000.0 / 2);
        launch_spin(n_sm, smem_dense, cal.glu_ns_row * rows / 1000.0 / 2);
        launch_spin(tiles * cal.tiles_d, smem_d, td);
        if (check && n > 0) {
            k_check<<<1, 64, 0, cs>>>(tab, tab + G, n, b.stride, d_err, 0, tpu);
        }
    }

    struct result {
        double t = 0;           // wall time of the whole run (s)
        double host_sync = 0;   // the graph thread's time in the blocks' synchronizations (s)
        double host_issue = 0;  // time in the copies' calls (wait + memcpy), whichever thread issues them (s)
        double host_launch = 0; // the graph thread's time in its kernel launches (s)
        int errors = 0;
        uint64_t copies = 0, in_place = 0, bytes = 0;
        size_t n_entries = 0, entry = 0;
    };

    result run(const config & c) {
        result res;
        plan p = make_plan(c);
        res.n_entries = p.n_entries;
        res.entry = p.entry;
        res.in_place = p.in_place;
        const int ng = blocks[0].nexp / c.G;
        // the check tables of every group, uploaded once (the real block uploads its addresses at the block)
        const size_t need = (size_t) p.n_groups * 2 * c.G;
        if (need > d_tabs_n) {
            if (d_tabs) {
                CK(cudaFree(d_tabs));
            }
            CK(cudaMalloc(&d_tabs, need * 8));
            d_tabs_n = need;
        }
        {
            std::vector<uint64_t> h(need, 0);
            for (int64_t g = 0; g < p.n_groups; g++) {
                std::copy(p.addr[g].begin(), p.addr[g].end(), h.begin() + g * 2 * c.G);
                std::copy(p.tag[g].begin(), p.tag[g].end(), h.begin() + g * 2 * c.G + c.G);
            }
            CK(cudaMemcpy(d_tabs, h.data(), need * 8, cudaMemcpyHostToDevice));
        }
        CK(cudaMemset(d_err, 0, sizeof(int)));
        // the ring as the cache left it: entries tagged with nothing valid
        CK(cudaMemset(ring, 0xff, ring_bytes));
        std::vector<cudaEvent_t> done(p.n_groups), copied(p.n_groups);
        for (int64_t g = 0; g < p.n_groups; g++) {
            CK(cudaEventCreateWithFlags(&done[g], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&copied[g], cudaEventDisableTiming));
        }
        CK(cudaDeviceSynchronize());

        const bool copies_on = c.copies;
        const bool dedicated = c.issuer == "thread" || c.issuer == "hthread";
        std::atomic<int64_t> recorded{0};         // groups whose done event is recorded
        std::atomic<int64_t> copied_rec{0};       // groups whose copies are all issued and their copied event recorded
        std::mutex mu;
        std::condition_variable cv;
        size_t next_op = 0;                       // graph-thread issuers: the first op not issued

        std::atomic<int64_t> issue_ns{0};
        int64_t last_wait = -1;   // the copy stream's last wait (wait1)
        auto issue = [&](const copy_op & op) {
            const auto i0 = std::chrono::steady_clock::now();
            struct at_end {
                std::chrono::steady_clock::time_point i0; std::atomic<int64_t> & acc;
                ~at_end() { acc += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - i0).count(); }
            } guard{i0, issue_ns};
            if (op.wait >= 0 && (!c.wait1 || op.wait > last_wait)) {
                if (c.issuer == "hthread") {
                    CK(cudaEventSynchronize(done[op.wait]));
                } else {
                    CK(cudaStreamWaitEvent(cp, done[op.wait], 0));
                }
                last_wait = op.wait;
            }
            CK(cudaMemcpyAsync(ring + (size_t) op.entry * p.entry, src_of(op), blocks[op.block].stride, cudaMemcpyHostToDevice, cp));
            res.copies++;
            res.bytes += blocks[op.block].stride;
        };
        // graph-thread issuers: the ops in order up to `limit` (block: the current block's end) whose wait is recorded
        auto advance = [&](size_t limit) {
            while (next_op < limit) {
                const copy_op & op = p.ops[next_op];
                if (op.wait >= recorded.load()) {
                    break;
                }
                issue(op);
                if ((int64_t) next_op == p.last_op_of_group[op.group]) {
                    CK(cudaEventRecord(copied[op.group], cp));
                    copied_rec.store(op.group + 1);
                }
                next_op++;
            }
        };
        std::thread th;
        if (copies_on && dedicated) {
            th = std::thread([&]() {
                CK(cudaSetDevice(0));
                for (size_t k = 0; k < p.ops.size(); k++) {
                    const copy_op & op = p.ops[k];
                    if (op.wait >= 0 && recorded.load() <= op.wait) {
                        std::unique_lock<std::mutex> lock(mu);
                        cv.wait(lock, [&]() { return recorded.load() > op.wait; });
                    }
                    issue(op);
                    if ((int64_t) k == p.last_op_of_group[op.group]) {
                        CK(cudaEventRecord(copied[op.group], cp));
                        {
                            std::lock_guard<std::mutex> lock(mu);
                            copied_rec.store(op.group + 1);
                        }
                        cv.notify_all();
                    }
                }
            });
        }

        std::vector<int32_t> hb(blocks[0].nexp + 1);
        int32_t * d_hb = nullptr;
        CK(cudaMalloc(&d_hb, hb.size() * 4));
        CK(cudaMemset(d_hb, 0, hb.size() * 4));
        double host_sync = 0, host_launch = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t bi = 0; bi < blocks.size(); bi++) {
            const block_rec & b = blocks[bi];
            if (c.compute) {
                const auto l0 = std::chrono::steady_clock::now();
                launch_dense(b);
                host_launch += std::chrono::duration<double>(std::chrono::steady_clock::now() - l0).count();
            }
            if (copies_on && !dedicated) {
                advance(c.issuer == "ahead" ? p.ops.size() : p.block_end[bi]);
            }
            if (c.sync) {
                CK(cudaMemcpyAsync(hb.data(), d_hb, hb.size() * 4, cudaMemcpyDeviceToHost, cs));
                const auto s0 = std::chrono::steady_clock::now();
                CK(cudaStreamSynchronize(cs));
                host_sync += std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
            }
            for (int gi = 0; gi < ng; gi++) {
                const int64_t g = (int64_t) bi * ng + gi;
                if (copies_on) {
                    if (!dedicated) {
                        advance(c.issuer == "ahead" ? p.ops.size() : p.block_end[bi]);
                        if (p.first_op_of_group[g] >= 0 && copied_rec.load() <= g) {
                            fprintf(stderr, "a group's copies wait for itself (group %lld)\n", (long long) g);
                            exit(1);
                        }
                    } else if (p.first_op_of_group[g] >= 0 && copied_rec.load() <= g) {
                        std::unique_lock<std::mutex> lock(mu);
                        cv.wait(lock, [&]() { return copied_rec.load() > g; });
                    }
                    if (p.first_op_of_group[g] >= 0) {
                        CK(cudaStreamWaitEvent(cs, copied[g], 0));
                    }
                }
                if (c.compute) {
                    const auto l0 = std::chrono::steady_clock::now();
                    launch_group(b, gi * c.G, c.G, p, g);
                    host_launch += std::chrono::duration<double>(std::chrono::steady_clock::now() - l0).count();
                }
                CK(cudaEventRecord(done[g], cs));
                {
                    std::lock_guard<std::mutex> lock(mu);
                    recorded.store(g + 1);
                }
                if (dedicated) {
                    cv.notify_all();
                } else if (copies_on) {
                    advance(c.issuer == "ahead" ? p.ops.size() : p.block_end[bi]);
                }
            }
        }
        if (th.joinable()) {
            th.join();
        }
        CK(cudaStreamSynchronize(cp));
        CK(cudaStreamSynchronize(cs));
        CK(cudaGetLastError());
        res.t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        res.host_sync = host_sync;
        res.host_launch = host_launch;
        res.host_issue = issue_ns.load() / 1e9;
        CK(cudaMemcpy(&res.errors, d_err, sizeof(int), cudaMemcpyDeviceToHost));
        for (int64_t g = 0; g < p.n_groups; g++) {
            CK(cudaEventDestroy(done[g]));
            CK(cudaEventDestroy(copied[g]));
        }
        CK(cudaFree(d_hb));
        return res;
    }

    // every copy back to back into the ring, no compute: the link's time for the run's bytes
    double copy_only(const config & c, uint64_t * bytes) {
        plan p = make_plan(c);
        CK(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t n = 0;
        for (const copy_op & op : p.ops) {
            CK(cudaMemcpyAsync(ring + (size_t) op.entry * p.entry, src_of(op), blocks[op.block].stride, cudaMemcpyHostToDevice, cp));
            n += blocks[op.block].stride;
        }
        CK(cudaStreamSynchronize(cp));
        *bytes = n;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> v;
    size_t a = 0;
    while (a <= s.size()) {
        const size_t b = s.find(',', a);
        v.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) {
            break;
        }
        a = b + 1;
    }
    return v;
}

static void usage() {
    printf(
        "ringsim [options]\n"
        "  --dump FILE        routing dumped by GGML_CUDA_MOE_RING_DUMP=FILE (else synthetic)\n"
        "  --merge K          K consecutive passes of each layer as one (K x the tokens: 3 x 9536 ~ 28416)\n"
        "  --passes N         prompt passes to run (default: all in the dump, synthetic 2)\n"
        "  --synth-tok N      synthetic: tokens per pass (9536); --synth-held F: share in the cache (0.2)\n"
        "  --held F           override the dump's cache: the F share of each layer's experts (the hottest) held\n"
        "  --G LIST           group sizes (32)\n"
        "  --ring LIST        ring sizes in MiB (900)\n"
        "  --issuer LIST      block (the real one), ahead, thread, hthread (block)\n"
        "  --sync LIST        1: bounds read back and the stream synchronized per block (real), 0: not (1)\n"
        "  --wait1 LIST       1: the copy stream waits once per group whose read it needs, 0: once per copy (real) (0)\n"
        "  --J N              MMQ tokens per tile (128)\n"
        "  --dense-ns-tok X   the rest of a layer per token (3600 ns: 34.3 ms at 9536)\n"
        "  --dense-kernels K  its launches (64)\n"
        "  --kscale X         every kernel time times X\n"
        "  --tgu T=US,..      MMQ gate/up us per 128x128 block by ggml type (22=59.7 44=60.7 43=80.7)\n"
        "  --td T=US,..       MMQ down us per block (20=21.7 42=21.4)\n"
        "  --smem-gu B --smem-d B   dynamic shared memory of the MMQ proxies (61952 57856: 1 block per SM on sm_89)\n"
        "  --no-check         no tag checks\n");
}

int main(int argc, char ** argv) {
    sim S;
    const char * dump = nullptr;
    int merge = 1, passes = -1, synth_tok = 9536;
    double synth_held = 0.2, held = -1;
    std::vector<std::string> Gs = {"32"}, rings = {"900"}, issuers = {"block"}, syncs = {"1"}, wait1s = {"0"};
    auto parse_map = [](const char * s, std::map<int, double> & m) {
        for (const std::string & kv : split(s)) {
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) {
                m[atoi(kv.substr(0, eq).c_str())] = atof(kv.substr(eq + 1).c_str());
            }
        }
    };
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage();
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--dump") dump = next();
        else if (a == "--merge") merge = atoi(next());
        else if (a == "--passes") passes = atoi(next());
        else if (a == "--synth-tok") synth_tok = atoi(next());
        else if (a == "--synth-held") synth_held = atof(next());
        else if (a == "--held") held = atof(next());
        else if (a == "--G") Gs = split(next());
        else if (a == "--ring") rings = split(next());
        else if (a == "--issuer") issuers = split(next());
        else if (a == "--sync") syncs = split(next());
        else if (a == "--wait1") wait1s = split(next());
        else if (a == "--J") S.cal.J = atoi(next());
        else if (a == "--dense-ns-tok") S.cal.dense_ns_tok = atof(next());
        else if (a == "--dense-kernels") S.cal.dense_kernels = atoi(next());
        else if (a == "--kscale") S.cal.kscale = atof(next());
        else if (a == "--tgu") parse_map(next(), S.cal.t_gu);
        else if (a == "--td") parse_map(next(), S.cal.t_d);
        else if (a == "--smem-gu") S.smem_gu = atoi(next());
        else if (a == "--smem-d") S.smem_d = atoi(next());
        else if (a == "--no-check") S.check = false;
        else if (a == "-v") S.verbose = true;
        else { usage(); return a == "--help" || a == "-h" ? 0 : 1; }
    }

    // the prompt
    std::vector<block_rec> all = dump ? load_dump(dump) : synthetic(passes > 0 ? passes : 2, synth_tok, synth_held, 1);
    // passes of the prompt: only the full ones (a pass of fewer tokens is the prompt's end)
    all = merge_passes(all, merge);
    int max_tok = 0;
    for (const block_rec & b : all) {
        max_tok = std::max(max_tok, b.ntok);
    }
    for (const block_rec & b : all) {
        if (b.ntok * 10 >= max_tok * 9) {
            S.blocks.push_back(b);
        }
    }
    {
        int pass = -1, last = INT32_MAX;
        std::vector<block_rec> kept;
        for (block_rec & b : S.blocks) {
            if (b.layer <= last) {
                pass++;
            }
            last = b.layer;
            b.pass = pass;
            if (passes <= 0 || pass < passes) {
                kept.push_back(b);
            }
        }
        S.blocks.swap(kept);
    }
    if (S.blocks.empty()) {
        fprintf(stderr, "no blocks\n");
        return 1;
    }
    if (held >= 0) {
        for (block_rec & b : S.blocks) {
            std::vector<int> order(b.nexp);
            for (int e = 0; e < b.nexp; e++) {
                order[e] = e;
            }
            std::sort(order.begin(), order.end(), [&](int x, int y) { return b.rows[x] > b.rows[y]; });
            b.held.assign(b.nexp, 0);
            for (int k = 0; k < (int) (held * b.nexp); k++) {
                b.held[order[k]] = 1;
            }
        }
    }
    uint64_t rows = 0, tok = 0, n_held = 0, n_exp = 0, tiles = 0;
    uint64_t tok_pass = 0;   // the prompt's tokens (one block per pass)
    for (const block_rec & b : S.blocks) {
        if (b.layer == S.blocks[0].layer) {
            tok_pass += b.ntok;
        }
    }
    std::map<std::pair<int, int>, int> types;
    for (const block_rec & b : S.blocks) {
        tok += b.ntok;
        for (int e = 0; e < b.nexp; e++) {
            rows += b.rows[e];
            n_held += b.held[e];
            tiles += (b.rows[e] + S.cal.J - 1) / S.cal.J;
        }
        n_exp += b.nexp;
        types[{b.tg, b.td}]++;
    }
    printf("prompt: %zu blocks (%d passes of %d tokens), %.1f rows per expert, MMQ columns %.2f x the rows (J = %d), %.1f %% of the experts in the cache\n",
           S.blocks.size(), S.blocks.back().pass + 1, S.blocks[0].ntok, (double) rows / n_exp, (double) tiles * S.cal.J / rows, S.cal.J,
           100.0 * n_held / n_exp);
    printf("layer types (gate/up, down: blocks):");
    for (const auto & t : types) {
        printf(" %d/%d: %d", t.first.first, t.first.second, t.second);
    }
    printf("\n");

    // the device
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    S.n_sm = prop.multiProcessorCount;
#if defined(__HIP_PLATFORM_AMD__)
    int khz = 0;
    CK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
    S.tpu = khz > 0 ? (uint64_t) khz / 1000 : 100;
    S.smem_gu = std::min<size_t>(S.smem_gu, prop.sharedMemPerBlock);
    S.smem_d = std::min<size_t>(S.smem_d, prop.sharedMemPerBlock);
#else
    CK(cudaFuncSetAttribute(k_spin, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) std::max(S.smem_gu, S.smem_d)));
#endif
    printf("device: %s, %d SMs; MMQ proxies %zu / %zu B of shared memory per block\n", prop.name, S.n_sm, S.smem_gu, S.smem_d);
    CK(cudaStreamCreateWithFlags(&S.cs, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&S.cp, cudaStreamNonBlocking));
    CK(cudaMalloc(&S.d_err, sizeof(int)));
    size_t largest = 0;
    for (const block_rec & b : S.blocks) {
        largest = std::max<size_t>(largest, b.stride);
    }
    int max_ring = 0;
    for (const std::string & r : rings) {
        max_ring = std::max(max_ring, atoi(r.c_str()));
    }
    S.ring_bytes = (size_t) max_ring * 1024 * 1024 + (1 << 20);
    CK(cudaMalloc(&S.ring, S.ring_bytes));
    // the source: more distinct tagged experts than the ring holds plus a layer (a stale entry never carries the tag)
    S.src_stride = (largest + 4095) / 4096 * 4096;
    S.n_src = (int) (S.ring_bytes / ((largest + 1024 + 255) / 256 * 256)) + 1031;
    {
        const size_t bytes = (size_t) S.n_src * S.src_stride;
        S.src = (char *) aligned_alloc(4096, bytes);
        if (!S.src) {
            fprintf(stderr, "cannot allocate %zu MiB of host memory\n", bytes >> 20);
            return 1;
        }
        memset(S.src, 0, bytes);
        std::set<uint64_t> strides;
        for (const block_rec & b : S.blocks) {
            strides.insert(b.stride);
        }
        for (int k = 0; k < S.n_src; k++) {
            // the tag at both ends of every stride a block may copy
            const uint64_t t = S.tag_of(k);
            memcpy(S.src + (size_t) k * S.src_stride, &t, 8);
            for (const uint64_t st : strides) {
                memcpy(S.src + (size_t) k * S.src_stride + st - 8, &t, 8);
            }
        }
        const auto r0 = std::chrono::steady_clock::now();
        CK(cudaHostRegister(S.src, bytes, cudaHostRegisterDefault));
        printf("source: %d experts of %zu KiB registered (%zu MiB, %.1f s)\n", S.n_src, S.src_stride >> 10, bytes >> 20,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count());
    }
    // the warm-up: launches and copies
    {
        config c;
        c.ring_mib = atoi(rings[0].c_str());
        uint64_t b = 0;
        S.copy_only(c, &b);
    }

    printf("\n%-4s %-6s %-6s %-9s %-4s | %8s %8s %8s %8s %7s | %8s %6s %7s %s\n", "G", "ring", "entr.", "issuer", "sync",
           "T (s)", "compute", "link", "exposed", "expo %", "t/s", "sync s", "GB", "check");
    std::map<std::string, double> comp_cache;
    uint64_t link_bytes = 0;
    double link = 0;
    {
        config c;
        c.ring_mib = atoi(rings[0].c_str());
        link = S.copy_only(c, &link_bytes);
    }
    for (const std::string & gs : Gs) {
        for (const std::string & rs : rings) {
            for (const std::string & ss : syncs) {
                config base;
                base.G = atoi(gs.c_str());
                base.ring_mib = atoi(rs.c_str());
                base.sync = atoi(ss.c_str());
                if (S.blocks[0].nexp % base.G != 0) {
                    continue;
                }
                // the compute alone (no copies) for this group size and synchronization
                const std::string key = gs + "/" + ss;
                if (!comp_cache.count(key)) {
                    config cc = base;
                    cc.copies = false;
                    comp_cache[key] = S.run(cc).t;
                }
                for (const std::string & is : issuers) for (const std::string & ws : wait1s) {
                    config c = base;
                    c.issuer = is;
                    c.wait1 = atoi(ws.c_str());
                    const sim::result r = S.run(c);
                    const double comp = comp_cache[key];
                    printf("%-4d %-6d %-6zu %-9s %-4d | %8.3f %8.3f %8.3f %8.3f %6.1f%% | %8.0f %6.3f %7.1f %s\n", c.G, c.ring_mib,
                           r.n_entries, (is + (c.wait1 ? "/w1" : "")).c_str(), c.sync, r.t, comp, link, r.t - comp, 100.0 * (r.t - comp) / r.t,
                           (double) tok_pass / r.t, r.host_sync, r.bytes / 1e9, r.errors ? "ERRORS" : "ok");
                    if (r.errors) {
                        printf("  %d tag errors\n", r.errors);
                    }
                    if (S.verbose) {
                        printf("  host: launches %.3f s, copy calls %.3f s (%.1f us a copy), synchronizations %.3f s\n", r.host_launch,
                               r.host_issue, 1e6 * r.host_issue / std::max<uint64_t>(1, r.copies), r.host_sync);
                    }
                    fflush(stdout);
                }
            }
        }
    }
    printf("\nT: the run; compute: the same without copies; link: the copies alone, back to back (%.1f GB, %.2f GB/s); exposed: T - compute\n",
           link_bytes / 1e9, link_bytes / 1e9 / link);
    CK(cudaHostUnregister(S.src));
    free(S.src);
    return 0;
}
