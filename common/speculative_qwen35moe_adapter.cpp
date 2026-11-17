#include "speculative_qwen35moe_adapter.h"

#include "../src/llama-model.h"

#include <array>
#include <algorithm>
#include <string_view>

namespace syclonith::speculative {
namespace {

bool read_architecture(const llama_model * model, std::string & architecture) {
    if (model == nullptr) {
        return false;
    }

    std::array<char, 128> value{};
    const int32_t n = llama_model_meta_val_str(model, "general.architecture", value.data(), value.size());
    if (n <= 0 || static_cast<size_t>(n) >= value.size()) {
        return false;
    }

    architecture.assign(value.data(), static_cast<size_t>(n));
    return true;
}

bool ends_with(const std::string & value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const ggml_tensor * find_tensor(
        const std::vector<std::pair<std::string, ggml_tensor *>> & tensors,
        const std::string & name) {
    const auto it = std::find_if(tensors.begin(), tensors.end(), [&](const auto & entry) {
        return entry.first == name;
    });
    return it == tensors.end() ? nullptr : it->second;
}

bool validate_norm_tensor(
        const ggml_tensor * tensor,
        int32_t hidden_size,
        const char * name,
        std::string & reason) {
    if (tensor == nullptr) {
        reason = std::string("missing required MTP tensor: ") + name;
        return false;
    }
    if (ggml_n_dims(tensor) != 1 || tensor->ne[0] != hidden_size) {
        reason = std::string("unexpected shape for MTP tensor: ") + name;
        return false;
    }
    return true;
}

} // namespace

bool Qwen35MoEMTPAdapter::inspect(
        const llama_model * target,
        const llama_model * draft,
        Qwen35MoEMTPAdapterInfo & info,
        std::string & reason) {
    info = {};
    reason.clear();

    std::string target_arch;
    std::string draft_arch;
    if (!read_architecture(target, target_arch) || !read_architecture(draft, draft_arch)) {
        reason = "could not read general.architecture from target or draft model metadata";
        return false;
    }
    if (target_arch != ARCHITECTURE || draft_arch != ARCHITECTURE) {
        reason = "target and draft must both use qwen35moe architecture";
        return false;
    }

    const int32_t target_hidden = llama_model_n_embd_out(target);
    const int32_t draft_hidden = llama_model_n_embd_out(draft);
    if (target_hidden <= 0 || target_hidden != draft_hidden) {
        reason = "target/draft output embedding widths do not match";
        return false;
    }

    const int32_t nextn_layers = llama_model_n_layer_nextn(draft);
    if (nextn_layers != 1) {
        reason = "Qwen35MoE adapter currently supports exactly one trained nextn prediction layer";
        return false;
    }

    const auto & tensors = llama_internal_get_tensor_map(draft);
    const std::string eh_suffix = ".nextn.eh_proj.weight";
    const auto eh_it = std::find_if(tensors.begin(), tensors.end(), [&](const auto & entry) {
        return ends_with(entry.first, eh_suffix);
    });
    if (eh_it == tensors.end()) {
        reason = "missing required MTP tensor suffix: .nextn.eh_proj.weight";
        return false;
    }

    const size_t marker = eh_it->first.rfind(".nextn.");
    if (marker == std::string::npos) {
        reason = "could not determine MTP tensor block prefix";
        return false;
    }
    const std::string prefix = eh_it->first.substr(0, marker + 1);
    const std::string eh_name = prefix + "nextn.eh_proj.weight";
    const std::string enorm_name = prefix + "nextn.enorm.weight";
    const std::string hnorm_name = prefix + "nextn.hnorm.weight";
    const std::string shared_norm_name = prefix + "nextn.shared_head_norm.weight";

    const ggml_tensor * eh_proj = find_tensor(tensors, eh_name);
    if (eh_proj == nullptr || ggml_n_dims(eh_proj) != 2) {
        reason = "missing or invalid rank for MTP projection tensor: " + eh_name;
        return false;
    }
    const bool projection_shape_ok =
        (eh_proj->ne[0] == target_hidden && eh_proj->ne[1] == 2 * target_hidden) ||
        (eh_proj->ne[1] == target_hidden && eh_proj->ne[0] == 2 * target_hidden);
    if (!projection_shape_ok) {
        reason = "unexpected dimensions for MTP projection tensor: " + eh_name;
        return false;
    }

    if (!validate_norm_tensor(find_tensor(tensors, enorm_name), target_hidden, enorm_name.c_str(), reason) ||
        !validate_norm_tensor(find_tensor(tensors, hnorm_name), target_hidden, hnorm_name.c_str(), reason) ||
        !validate_norm_tensor(find_tensor(tensors, shared_norm_name), target_hidden, shared_norm_name.c_str(), reason)) {
        return false;
    }

    info.architecture = draft_arch;
    info.hidden_size = target_hidden;
    info.nextn_layers = nextn_layers;
    info.tensor_prefix = prefix;
    info.required_tensors = { eh_name, enorm_name, hnorm_name, shared_norm_name };
    return true;
}

} // namespace syclonith::speculative
