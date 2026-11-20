// the prompt's expert ring (moe_cache_ring_zone, issue 27): a clear (base == NULL) on a device whose ring was
// never set up must not create ring state. The ring's copy stream is a compute HW queue; one past the gfx1030
// runlist's eight oversubscribes it and the prefill's copy sync hangs. Before the guard, the first such clear
// created the stream and the process's queue count grew, which this test detects.
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-moe-cache.h"

#include <cstdio>

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// the process's compute queues on KFD (type 0); -1 when the KFD is not there (not Linux/ROCm, e.g.
// a Linux+NVIDIA host), so that a pass without visibility is reported, not asserted as 0 -> 0
static int kfd_compute_queues(void) {
#if defined(__linux__)
    if (access("/sys/class/kfd/kfd", F_OK) != 0) {
        return -1;
    }
    char dir[64];
    snprintf(dir, sizeof(dir), "/sys/class/kfd/kfd/proc/%d/queues", getpid());
    DIR * d = opendir(dir);
    if (!d) {
        return 0;   // the process has opened no queue
    }
    int n = 0;
    struct dirent * e;
    while ((e = readdir(d)) != nullptr) {
        if (e->d_name[0] == '.') {
            continue;
        }
        char type[128];
        snprintf(type, sizeof(type), "%s/%s/type", dir, e->d_name);
        int fd = open(type, O_RDONLY);
        if (fd < 0) {
            continue;
        }
        char buf[8] = {0};
        if (read(fd, buf, sizeof(buf) - 1) > 0 && buf[0] == '0') {
            n++;
        }
        close(fd);
    }
    closedir(d);
    return n;
#else
    return -1;
#endif
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        const ggml_backend_dev_t cand = ggml_backend_dev_get(i);
        const int t = ggml_backend_dev_type(cand);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU || t == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            dev = cand;
            break;
        }
    }
    if (!dev) {
        printf("SKIP: no GPU device\n");
        return 0;
    }
    if (!ggml_moe_cache.session_create || !ggml_moe_cache.ring_zone || !ggml_moe_cache.session_destroy) {
        printf("SKIP: no moe cache provider (CPU-only build)\n");
        return 0;
    }
    printf("test-moe-ringzone: %s\n", ggml_backend_dev_name(dev));

    const ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fflush(stdout);
        fprintf(stderr, "FAIL: backend init for %s\n", ggml_backend_dev_name(dev));
        return 1;
    }
    void * const backends[1] = { backend };
    void * session = ggml_moe_cache.session_create(backends, 1, nullptr);
    if (!session) {
        fflush(stdout);
        fprintf(stderr, "FAIL: session_create returned null (the moe cache is off for this device? GGML_CUDA_MOE_CACHE=1)\n");
        return 1;
    }

    // the baseline after the session, so the queues the session itself opens do not count against the clear
    const int before = kfd_compute_queues();

    // the prefetch_restore clear path on a device whose ring was never set up
    ggml_moe_cache.ring_zone(session, backend, nullptr, 0);
    int after = kfd_compute_queues();
    if (before >= 0 && after != before) {
        fflush(stdout);
        fprintf(stderr, "FAIL: the clear created ring state: %d -> %d compute queues (the ring's copy stream is a\n"
                "HW queue; on a gfx1030 one past the runlist's eight oversubscribes it and the prefill's copy\n"
                "hangs, issue 27)\n", before, after);
        return 1;
    }

    // a second clear must not create anything either
    ggml_moe_cache.ring_zone(session, backend, nullptr, 0);
    after = kfd_compute_queues();
    if (before >= 0 && after != before) {
        fflush(stdout);
        fprintf(stderr, "FAIL: a second clear grew the queue count: %d -> %d\n", before, after);
        return 1;
    }
    if (before < 0) {
        printf("note: no KFD queue visibility (not Linux/ROCm); the calls ran, the count is not asserted\n");
    }

    ggml_moe_cache.session_destroy(session);
    ggml_backend_free(backend);
    printf("ok: a ring_zone clear of a never set up ring creates no ring state\n");
    return 0;
}
