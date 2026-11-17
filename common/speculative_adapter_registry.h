#pragma once

#include "llama.h"

#include <cstdint>
#include <string>

namespace syclonith::speculative {

// This registry is deliberately a discovery/validation bridge. It does not
// claim that an architecture's graph execution has migrated to ArchitectureAdapter.
struct AdapterInspection {
    int32_t hidden_size = 0;
    int32_t prediction_layers = 0;
    uint32_t validated_tensor_count = 0;
};

using AdapterInspectFn = bool (*)(
        const llama_model * target,
        const llama_model * draft,
        AdapterInspection & inspection,
        std::string & reason);

struct AdapterDescriptor {
    const char * id;
    const char * architecture;
    AdapterInspectFn inspect;
};

enum class AdapterSelectionStatus {
    no_registered_adapter,
    selected,
    rejected,
    invalid_model_metadata,
};

struct AdapterSelection {
    AdapterSelectionStatus status = AdapterSelectionStatus::no_registered_adapter;
    const AdapterDescriptor * adapter = nullptr;
    AdapterInspection inspection;
    std::string architecture;
    std::string reason;

    bool selected() const {
        return status == AdapterSelectionStatus::selected && adapter != nullptr;
    }
};

// Explicit, deterministic built-in registry. New architecture adapters must be
// added here; registration order and static initialisation are not observable.
const AdapterDescriptor * find_adapter(const std::string & architecture);

// Resolves the target architecture and validates the target/draft pair.
// Unknown architectures and rejected pairs leave the existing MTP executor
// available to the caller as a compatibility fallback.
AdapterSelection select_adapter(
        const llama_model * target,
        const llama_model * draft);

} // namespace syclonith::speculative
