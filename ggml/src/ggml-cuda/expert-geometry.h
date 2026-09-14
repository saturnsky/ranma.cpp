#pragma once

// Pure description of a MoE expert tensor layout. This header is deliberately
// free of CUDA/HIP includes so the policy code and its tests build in a
// CPU-only tree.

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

struct geometry {
    int n_layers  = 0;                // routed layer index span = max routed layer + 1
    int n_experts = 0;
    static constexpr int n_kinds = 3; // 0 = up, 1 = gate, 2 = down

    std::vector<std::array<size_t, 3>> nb2;                  // [layer][kind] bytes per expert; 0 when the layer is not routed
    std::vector<int>                   layer_class;          // [layer] size class index or -1
    std::vector<std::array<const ggml_tensor *, 3>> tensors; // [layer][kind], nullptr when not routed
    std::vector<std::array<size_t, 3>> class_bytes;          // [class][kind]
    std::vector<int>                   class_layers;         // [class] number of layers in the class

    size_t n_counts() const {
        return (size_t) n_layers*(size_t) n_experts;
    }

    size_t class_total_bytes(int cls) const {
        size_t result = 0;
        for (int kind = 0; kind < n_kinds; ++kind) {
            result += class_bytes[cls][kind];
        }
        return result;
    }

    int n_routed_layers() const {
        int result = 0;
        for (int layer = 0; layer < n_layers; ++layer) {
            if (layer_class[layer] >= 0) {
                ++result;
            }
        }
        return result;
    }

    // Compatibility key for a stored profile: anything that would change the
    // meaning of a per-(layer, expert) score has to change this string.
    std::string signature() const {
        std::string text = "ranma-expert-geometry-v1\n";
        char line[256];
        snprintf(line, sizeof(line), "%d,%d\n", n_layers, n_experts);
        text += line;
        for (int layer = 0; layer < n_layers; ++layer) {
            if (layer_class[layer] < 0) {
                continue;
            }
            for (int kind = 0; kind < n_kinds; ++kind) {
                const ggml_tensor * tensor = tensors[layer][kind];
                snprintf(line, sizeof(line), "%d,%d,%d,%lld,%lld,%llu\n",
                    layer, kind, (int) tensor->type,
                    (long long) tensor->ne[0], (long long) tensor->ne[1],
                    (unsigned long long) nb2[layer][kind]);
                text += line;
            }
        }
        return text;
    }
};

// Accepts "blk.<layer>.ffn_{up,gate,down}_exps" with any suffix. Same matching
// rule as the reference implementation, minus its global n_layers upper bound (this
// header has no global model state).
inline bool parse_expert_tensor_name(const char * name, int & layer, int & kind) {
    if (name == nullptr || strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    char * end = nullptr;
    const long parsed_layer = strtol(name + 4, &end, 10);
    if (end == name + 4 || parsed_layer < 0) {
        return false;
    }
    if (strstr(end, ".ffn_up_exps") != nullptr) {
        kind = 0;
    } else if (strstr(end, ".ffn_gate_exps") != nullptr) {
        kind = 1;
    } else if (strstr(end, ".ffn_down_exps") != nullptr) {
        kind = 2;
    } else {
        return false;
    }
    layer = (int) parsed_layer;
    return true;
}

inline bool build_geometry(const ggml_context * ctx, geometry & out, std::string & reason) {
    out = geometry();
    reason.clear();

    struct found {
        int layer = -1;
        int kind  = -1;
        const ggml_tensor * tensor = nullptr;
    };
    std::vector<found> routed;
    int max_layer = -1;
    int64_t experts = -1;

    char message[512];
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != nullptr;
            tensor = ggml_get_next_tensor(ctx, tensor)) {
        int layer = -1;
        int kind  = -1;
        if (!parse_expert_tensor_name(ggml_get_name(tensor), layer, kind)) {
            continue;
        }
        if (tensor->view_src != nullptr) {
            snprintf(message, sizeof(message), "expert tensor '%s' is a view", ggml_get_name(tensor));
            reason = message;
            return false;
        }
        if (!ggml_is_quantized(tensor->type)) {
            snprintf(message, sizeof(message), "expert tensor '%s' is not quantized", ggml_get_name(tensor));
            reason = message;
            return false;
        }
        if (!ggml_is_contiguous(tensor)) {
            snprintf(message, sizeof(message), "expert tensor '%s' is not contiguous", ggml_get_name(tensor));
            reason = message;
            return false;
        }
        if (tensor->ne[3] != 1) {
            snprintf(message, sizeof(message), "expert tensor '%s' has ne[3] = %lld", ggml_get_name(tensor),
                (long long) tensor->ne[3]);
            reason = message;
            return false;
        }
        if (experts < 0) {
            experts = tensor->ne[2];
        } else if (experts != tensor->ne[2]) {
            snprintf(message, sizeof(message), "expert tensor '%s' has %lld experts, expected %lld",
                ggml_get_name(tensor), (long long) tensor->ne[2], (long long) experts);
            reason = message;
            return false;
        }
        for (const found & other : routed) {
            if (other.layer == layer && other.kind == kind) {
                snprintf(message, sizeof(message), "duplicate expert tensor for layer %d kind %d", layer, kind);
                reason = message;
                return false;
            }
        }
        routed.push_back({layer, kind, tensor});
        max_layer = std::max(max_layer, layer);
    }

    if (routed.empty()) {
        // No MoE in this context: a valid answer, not a failure.
        return true;
    }
    if (experts <= 0) {
        reason = "expert tensors have a non-positive expert count";
        return false;
    }

    out.n_layers  = max_layer + 1;
    out.n_experts = (int) experts;
    out.nb2.assign(out.n_layers, {0, 0, 0});
    out.layer_class.assign(out.n_layers, -1);
    out.tensors.assign(out.n_layers, {nullptr, nullptr, nullptr});
    for (const found & item : routed) {
        out.tensors[item.layer][item.kind] = item.tensor;
        out.nb2[item.layer][item.kind]     = item.tensor->nb[2];
    }
    for (int layer = 0; layer < out.n_layers; ++layer) {
        int present = 0;
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            present += out.tensors[layer][kind] != nullptr ? 1 : 0;
        }
        if (present != 0 && present != geometry::n_kinds) {
            snprintf(message, sizeof(message), "routed layer %d is missing one of up/gate/down", layer);
            reason = message;
            return false;
        }
    }

    // Size classes. The reference implementation compared kind bytes plus type/ne0/ne1 of a reference
    // layer; generalized here to compare (type, ne0, ne1, nb2) per kind, which
    // is the same predicate expressed without a reference-layer lookup and
    // without relying on the byte size alone.
    std::vector<int> reference;
    for (int layer = 0; layer < out.n_layers; ++layer) {
        if (out.tensors[layer][0] == nullptr) {
            continue;
        }
        int cls = -1;
        for (int i = 0; i < (int) reference.size(); ++i) {
            const int ref = reference[i];
            bool same = true;
            for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                const ggml_tensor * a = out.tensors[ref][kind];
                const ggml_tensor * b = out.tensors[layer][kind];
                same = same && a->type == b->type && a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] &&
                    out.nb2[ref][kind] == out.nb2[layer][kind];
            }
            if (same) {
                cls = i;
                break;
            }
        }
        if (cls < 0) {
            reference.push_back(layer);
            out.class_bytes.push_back(out.nb2[layer]);
            out.class_layers.push_back(0);
            cls = (int) reference.size() - 1;
        }
        out.layer_class[layer] = cls;
        ++out.class_layers[cls];
    }
    return true;
}

} // namespace ggml_cuda_expert
