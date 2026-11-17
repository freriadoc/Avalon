#include "speculative_adapter_registry.h"

#include "speculative_qwen35moe_adapter.h"

#include <array>
#include <string>

namespace syclonith::speculative {
namespace {

bool read_architecture(const llama_model * model, std::string & architecture) {
    if (model == nullptr) {
        return false;
    }

    std::array<char, 128> value{};
    const int32_t n = llama_model_meta_val_str(
            model, "general.architecture", value.data(), value.size());
    if (n <= 0 || static_cast<size_t>(n) >= value.size()) {
        return false;
    }

    architecture.assign(value.data(), static_cast<size_t>(n));
    return true;
}

bool inspect_qwen35moe(
        const llama_model * target,
        const llama_model * draft,
        AdapterInspection & inspection,
        std::string & reason) {
    Qwen35MoEMTPAdapterInfo qwen_info;
    if (!Qwen35MoEMTPAdapter::inspect(target, draft, qwen_info, reason)) {
        return false;
    }

    inspection.hidden_size = qwen_info.hidden_size;
    inspection.prediction_layers = qwen_info.nextn_layers;
    inspection.validated_tensor_count =
            static_cast<uint32_t>(qwen_info.required_tensors.size());
    return true;
}

const std::array<AdapterDescriptor, 1> & built_in_adapters() {
    static const std::array<AdapterDescriptor, 1> adapters = {{
        { Qwen35MoEMTPAdapter::ID, Qwen35MoEMTPAdapter::ARCHITECTURE, inspect_qwen35moe },
    }};
    return adapters;
}

} // namespace

const AdapterDescriptor * find_adapter(const std::string & architecture) {
    for (const auto & adapter : built_in_adapters()) {
        if (architecture == adapter.architecture) {
            return &adapter;
        }
    }
    return nullptr;
}

AdapterSelection select_adapter(
        const llama_model * target,
        const llama_model * draft) {
    AdapterSelection result;
    if (!read_architecture(target, result.architecture)) {
        result.status = AdapterSelectionStatus::invalid_model_metadata;
        result.reason = "could not read target general.architecture metadata";
        return result;
    }

    result.adapter = find_adapter(result.architecture);
    if (result.adapter == nullptr) {
        result.status = AdapterSelectionStatus::no_registered_adapter;
        result.reason = "no architecture adapter is registered for " + result.architecture;
        return result;
    }

    if (!result.adapter->inspect(target, draft, result.inspection, result.reason)) {
        result.status = AdapterSelectionStatus::rejected;
        return result;
    }

    result.status = AdapterSelectionStatus::selected;
    return result;
}

} // namespace syclonith::speculative
