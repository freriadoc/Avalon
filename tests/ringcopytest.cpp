// Banc des copies hote -> VRAM depuis une memoire mappee (mmap d'un fichier du modele) :
//   direct : hipMemcpyAsync depuis la memoire paginable (ROCm verrouille les pages lui-meme)
//   buun   : tampon verrouille unique de 1 Mio, synchronisation a chaque morceau
//   anneau : N tampons verrouilles de S octets ; un tampon n'est reutilise qu'apres l'evenement
//            qui marque la fin de sa copie precedente ; l'appel rend la main sans attendre la fin
// RING_NT=1 : l'anneau copie avec copy_nt (hors cache) au lieu de memcpy.
// RING_WC=1 : tampons de l'anneau en ecriture combinee (hipHostMallocWriteCombined).
// Chaque variante copie la meme region (pages deja en RAM), relit la VRAM et la compare.
// build: hipcc -O2 -std=c++17 tests/ringcopytest.cpp -o /tmp/ringcopytest
// usage: ringcopytest FICHIER [Mio=2048] [repetitions=3] [anneaux, ex. 2x4,3x16,2x0.5]
#include <hip/hip_runtime.h>
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(1); } } while (0)

// copie hors cache : ecritures non temporelles (la destination ne passe pas par le cache) ;
// dst aligne sur 32 octets ; la barriere rend les ecritures visibles avant le lancement du DMA
__attribute__((target("avx2")))
static void copy_nt(void * dst, const void * src, size_t n) {
    char * d = (char *) dst; const char * s = (const char *) src;
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(s + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(s + i + 32));
        const __m256i c = _mm256_loadu_si256((const __m256i *)(s + i + 64));
        const __m256i e = _mm256_loadu_si256((const __m256i *)(s + i + 96));
        _mm256_stream_si256((__m256i *)(d + i), a);
        _mm256_stream_si256((__m256i *)(d + i + 32), b);
        _mm256_stream_si256((__m256i *)(d + i + 64), c);
        _mm256_stream_si256((__m256i *)(d + i + 96), e);
    }
    memcpy(d + i, s + i, n - i);
    _mm_sfence();
}
static const bool use_nt = getenv("RING_NT") != nullptr;

struct ring {
    std::vector<void *>      buf;
    std::vector<hipEvent_t>  ev;
    size_t                   slot_size;
    size_t                   next = 0;
    ring(size_t n, size_t size) : buf(n), ev(n), slot_size(size) {
        for (size_t i = 0; i < n; i++) {
            CHECK(hipHostMalloc(&buf[i], size, hipHostMallocPortable | (getenv("RING_WC") ? hipHostMallocWriteCombined : 0)));
            CHECK(hipEventCreateWithFlags(&ev[i], hipEventDisableTiming));
            CHECK(hipEventRecord(ev[i], nullptr)); // deja « libre »
        }
    }
    ~ring() {
        for (size_t i = 0; i < buf.size(); i++) { hipEventDestroy(ev[i]); hipHostFree(buf[i]); }
    }
    // copie hote -> VRAM ; la source peut etre modifiee des le retour
    void copy_h2d(void * dst, const void * src, size_t size, hipStream_t stream) {
        for (size_t off = 0; off < size; off += slot_size) {
            const size_t n = size - off < slot_size ? size - off : slot_size;
            CHECK(hipEventSynchronize(ev[next]));
            if (use_nt) copy_nt(buf[next], (const char *) src + off, n);
            else        memcpy(buf[next], (const char *) src + off, n);
            CHECK(hipMemcpyAsync((char *) dst + off, buf[next], n, hipMemcpyHostToDevice, stream));
            CHECK(hipEventRecord(ev[next], stream));
            next = (next + 1) % buf.size();
        }
    }
};

static void copy_buun(void * dst, const void * src, size_t size, hipStream_t stream, void * stage) {
    const size_t chunk = 1 << 20;
    for (size_t off = 0; off < size; off += chunk) {
        const size_t n = size - off < chunk ? size - off : chunk;
        memcpy(stage, (const char *) src + off, n);
        CHECK(hipMemcpyAsync((char *) dst + off, stage, n, hipMemcpyHostToDevice, stream));
        CHECK(hipStreamSynchronize(stream));
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s FICHIER [Mio] [repetitions]\n", argv[0]); return 1; }
    const int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t size = (size_t)(argc > 2 ? atol(argv[2]) : 2048) << 20;
    const int reps = argc > 3 ? atoi(argv[3]) : 3;
    if (size > (size_t) st.st_size) size = st.st_size & ~(size_t) 4095;
    const char * src = (const char *) mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (src == MAP_FAILED) { perror("mmap"); return 1; }
    // pages en RAM avant de mesurer (c'est le cas du prefill : les experts sont deja lus)
    volatile unsigned long sum = 0;
    for (size_t i = 0; i < size; i += 4096) sum += src[i];

    // la copie est decoupee comme celle d'un tenseur d'experts : blocs de 450 Mio
    const size_t tensor = 450u << 20;
    char * dev; CHECK(hipMalloc(&dev, size));
    std::vector<char> back(size);
    hipStream_t stream; CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    void * stage; CHECK(hipHostMalloc(&stage, 1 << 20, hipHostMallocPortable));

    // variante : n tampons de s octets (n = 0 : direct si s = 0, buun sinon)
    struct variant { std::string name; size_t n, s; };
    std::vector<variant> variants = { {"direct", 0, 0}, {"buun 1 Mio sync", 0, 1} };
    // anneaux : argument 4, liste « NxMIO,... » (Mio decimaux), ex. 2x4,3x2,2x0.5
    const char * list = argc > 4 ? argv[4] : "2x4,4x4,4x8,8x8,2x16,3x16,4x16,8x16,2x32,4x32";
    for (const char * p = list; *p; ) {
        char * end; const size_t n = strtoul(p, &end, 10);
        const double mib = strtod(end + 1, &end);
        char name[64]; snprintf(name, sizeof name, "anneau %zu x %g Mio", n, mib);
        variants.push_back({name, n, (size_t)(mib * 1048576.0)});
        p = *end == ',' ? end + 1 : end;
    }
    for (const variant & v : variants) {
        ring * r = v.n ? new ring(v.n, v.s) : nullptr;
        double best = 0;
        for (int rep = 0; rep < reps; rep++) {
            CHECK(hipMemsetAsync(dev, 0, size, stream));
            CHECK(hipStreamSynchronize(stream));
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t off = 0; off < size; off += tensor) {
                const size_t n = size - off < tensor ? size - off : tensor;
                if (r)           r->copy_h2d(dev + off, src + off, n, stream);
                else if (v.s)    copy_buun(dev + off, src + off, n, stream, stage);
                else             CHECK(hipMemcpyAsync(dev + off, src + off, n, hipMemcpyHostToDevice, stream));
            }
            CHECK(hipStreamSynchronize(stream));
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (size / s > best) best = size / s;
            CHECK(hipMemcpy(back.data(), dev, size, hipMemcpyDeviceToHost));
            if (memcmp(back.data(), src, size) != 0) { printf("%-22s CONTENU FAUX (rep %d)\n", v.name.c_str(), rep); return 1; }
        }
        printf("%-22s %6.2f Go/s (meilleure de %d, %zu Mio, contenu relu identique)\n", v.name.c_str(), best / 1e9, reps, size >> 20);
        delete r;
    }
    return 0;
}
