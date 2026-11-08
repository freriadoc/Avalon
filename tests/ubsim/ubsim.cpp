// ubsim : the prefill graph's compute buffer at several ubatch sizes, without the weights (llama_model_params.no_alloc:
// metadata only, nothing allocated on the devices) - a measuring context per size, as --fit makes them. Each context
// reserves its worst-case graphs and ggml-alloc records every block (size, birth and death events, op, type, name) in
// GGML_ALLOC_PLAN_DUMP; tests/ubsim/ubplan.c places them offline as the borrowed compute buffer would be (chunk cap),
// and attributes the peak to the tensors that make it.
//
//   UBSIM_SIZES=8192,9536,12288 UBSIM_DIR=out ubsim <the server's arguments: -m, -c, -fa, -ctk, -ctv, ...>
//
// build (HIP tree): clang++ -O2 -std=c++17 -I include -I ggml/include -I common -I vendor tests/ubsim/ubsim.cpp -o ubsim
//                   -L build-hip/bin -lllama -lllama-common -lggml -lggml-base -Wl,-rpath,$PWD/build-hip/bin
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    const char * sz = getenv("UBSIM_SIZES");
    const std::string dir = getenv("UBSIM_DIR") ? getenv("UBSIM_DIR") : ".";
    std::vector<int> sizes;
    {
        std::stringstream ss(sz ? sz : "9536");
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) {
                sizes.push_back(atoi(tok.c_str()));
            }
        }
    }
    llama_backend_init();
    llama_model_params mp = common_model_params_to_llama(params);
    mp.no_alloc  = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;   // as common/fit.cpp: nothing mapped
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "ubsim: failed to load %s\n", params.model.path.c_str());
        return 1;
    }
    for (const int T : sizes) {
        const std::string dump = dir + "/dump-" + std::to_string(T) + ".txt";
        remove(dump.c_str());
        setenv("GGML_ALLOC_PLAN_DUMP", dump.c_str(), 1);
        llama_context_params cp = common_context_params_to_llama(params);
        cp.n_batch  = std::max<uint32_t>(cp.n_batch, T);
        cp.n_ubatch = T;
        // the server's output limits (server_output_limits): a prompt outputs its last token, an MTP verification a
        // few; UBSIM_OUTPUTS overrides (default 4)
        cp.n_outputs_max         = getenv("UBSIM_OUTPUTS") ? atoi(getenv("UBSIM_OUTPUTS")) : 4;
        cp.n_outputs_max_per_seq = cp.n_outputs_max;
        fprintf(stderr, "ubsim: ===== n_ubatch %d =====\n", T);
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            fprintf(stderr, "ubsim: no context at n_ubatch %d\n", T);
            continue;
        }
        llama_free(ctx);
        unsetenv("GGML_ALLOC_PLAN_DUMP");
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
