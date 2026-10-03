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

// Per expert rows that a bank's kernels read next to the weight, by expert id: the rot_in / rot_out of an EXL3 bank,
// one row of nb[1] bytes per expert. The VRAM tier keeps them in arenas of their own, at the slot of the weight.
using expert_aux_tensors = std::array<std::array<const ggml_tensor *, 2>, 3>; // [kind][0 = rot_in, 1 = rot_out]
using expert_aux_bytes   = std::array<std::array<size_t, 2>, 3>;              // [kind][aux] bytes per expert, 0 = none

struct geometry {
    int n_layers  = 0;                // routed layer index span = max routed layer + 1
    int n_experts = 0;
    static constexpr int n_kinds = 3; // 0 = up, 1 = gate, 2 = down
    static constexpr int n_aux   = 2; // 0 = rot_in, 1 = rot_out

    std::vector<std::array<size_t, 3>> nb2;                  // [layer][kind] bytes per expert; 0 when the layer is not routed
    std::vector<int>                   layer_class;          // [layer] size class index or -1
    std::vector<std::array<const ggml_tensor *, 3>> tensors; // [layer][kind], nullptr when not routed
    std::vector<std::array<size_t, 3>> class_bytes;          // [class][kind]
    std::vector<int>                   class_layers;         // [class] number of layers in the class
    // aux rows (EXL3 rotations); empty, or null / 0 entries, for other banks and for geometries built by hand
    std::vector<expert_aux_tensors>    aux;                  // [layer]
    std::vector<expert_aux_bytes>      aux_nb;               // [layer]
    std::vector<expert_aux_bytes>      class_aux;            // [class]

    size_t n_counts() const {
        return (size_t) n_layers*(size_t) n_experts;
    }

    // Weight bytes of one expert of the class, all kinds: what the host and file tiers hold.
    size_t class_total_bytes(int cls) const {
        size_t result = 0;
        for (int kind = 0; kind < n_kinds; ++kind) {
            result += class_bytes[cls][kind];
        }
        return result;
    }

    size_t class_aux_bytes(int cls, int kind, int which) const {
        return cls >= 0 && (size_t) cls < class_aux.size() ? class_aux[cls][kind][which] : 0;
    }

    size_t class_aux_total_bytes(int cls) const {
        size_t result = 0;
        for (int kind = 0; kind < n_kinds; ++kind) {
            for (int which = 0; which < n_aux; ++which) {
                result += class_aux_bytes(cls, kind, which);
            }
        }
        return result;
    }

    // VRAM bytes of one resident expert of the class: its weights and the aux rows that move with them. Equal to
    // class_total_bytes for a bank without aux rows.
    size_t class_vram_bytes(int cls) const {
        return class_total_bytes(cls) + class_aux_total_bytes(cls);
    }

    const ggml_tensor * aux_tensor(int layer, int kind, int which) const {
        return layer >= 0 && (size_t) layer < aux.size() ? aux[layer][kind][which] : nullptr;
    }

    bool has_aux() const {
        for (size_t cls = 0; cls < class_aux.size(); ++cls) {
            if (class_aux_total_bytes((int) cls) != 0) {
                return true;
            }
        }
        return false;
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
    // meaning of a per-(layer, expert) score has to change this string. The aux
    // rows are not in it: they change what a resident expert costs, not what a
    // score counts, and a bank that has them (EXL3) already differs by its type.
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

// Accepts "blk.<layer>.ffn_{up,gate,down}_exps.weight" only: other tensors of a bank (.bias, .scale, the
// .rot_in / .rot_out of EXL3) are not expert weights. No global n_layers upper bound (this header has no
// global model state).
inline bool parse_expert_tensor_name(const char * name, int & layer, int & kind) {
    if (name == nullptr || strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    char * end = nullptr;
    const long parsed_layer = strtol(name + 4, &end, 10);
    if (end == name + 4 || parsed_layer < 0) {
        return false;
    }
    if (strcmp(end, ".ffn_up_exps.weight") == 0) {
        kind = 0;
    } else if (strcmp(end, ".ffn_gate_exps.weight") == 0) {
        kind = 1;
    } else if (strcmp(end, ".ffn_down_exps.weight") == 0) {
        kind = 2;
    } else {
        return false;
    }
    layer = (int) parsed_layer;
    return true;
}

// Accepts "blk.<layer>.ffn_{up,gate,down}_exps.rot_{in,out}": the aux rows of an EXL3 bank.
inline bool parse_expert_aux_name(const char * name, int & layer, int & kind, int & which) {
    if (name == nullptr || strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    char * end = nullptr;
    const long parsed_layer = strtol(name + 4, &end, 10);
    if (end == name + 4 || parsed_layer < 0) {
        return false;
    }
    static const char * const kinds[3] = { ".ffn_up_exps.", ".ffn_gate_exps.", ".ffn_down_exps." };
    static const char * const auxes[2] = { "rot_in", "rot_out" };
    for (int k = 0; k < 3; ++k) {
        const size_t n = strlen(kinds[k]);
        if (strncmp(end, kinds[k], n) != 0) {
            continue;
        }
        for (int a = 0; a < 2; ++a) {
            if (strcmp(end + n, auxes[a]) == 0) {
                layer = (int) parsed_layer;
                kind  = k;
                which = a;
                return true;
            }
        }
    }
    return false;
}

// An aux tensor the VRAM tier can move with its expert: F16 or F32 rows, one per expert of the bank, packed with the
// stride nb[1]. Anything else stays where it is and is read by expert id, as without the cache.
inline bool expert_aux_usable(const ggml_tensor * t, int64_t experts) {
    return t != nullptr && t->view_src == nullptr && (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32) &&
        t->ne[1] == experts && t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t) && t->nb[1] != 0;
}

// Size classes. The reference implementation compared kind bytes plus type/ne0/ne1 of a reference
// layer; generalized here to compare (type, ne0, ne1, nb2) per kind, which is the same predicate
// expressed without a reference-layer lookup and without relying on the byte size alone. Classes
// are numbered in the order of their first layer.
inline void assign_size_classes(geometry & out) {
    out.class_bytes.clear();
    out.class_layers.clear();
    out.class_aux.clear();
    out.layer_class.assign(out.n_layers, -1);
    // the aux rows are part of a layer's slice layout: two layers share a class only with the same aux strides
    auto aux_of = [&](int layer) {
        return (size_t) layer < out.aux_nb.size() ? out.aux_nb[layer] : expert_aux_bytes{};
    };
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
            same = same && aux_of(ref) == aux_of(layer);
            if (same) {
                cls = i;
                break;
            }
        }
        if (cls < 0) {
            reference.push_back(layer);
            out.class_bytes.push_back(out.nb2[layer]);
            out.class_aux.push_back(aux_of(layer));
            out.class_layers.push_back(0);
            cls = (int) reference.size() - 1;
        }
        out.layer_class[layer] = cls;
        ++out.class_layers[cls];
    }
}

// Every tensor of `tensors` that is a routed expert weight is part of the geometry; the others are
// ignored. build_geometry below is the same over the tensors of one context.
inline bool build_geometry_tensors(const std::vector<const ggml_tensor *> & tensors, geometry & out, std::string & reason) {
    out = geometry();
    reason.clear();

    struct found {
        int layer = -1;
        int kind  = -1;
        const ggml_tensor * tensor = nullptr;
    };
    std::vector<found> routed;
    struct found_aux {
        int layer = -1, kind = -1, which = -1;
        const ggml_tensor * tensor = nullptr;
    };
    std::vector<found_aux> auxes;
    int max_layer = -1;
    int64_t experts = -1;

    char message[512];
    for (const ggml_tensor * tensor : tensors) {
        int layer = -1;
        int kind  = -1;
        if (!parse_expert_tensor_name(ggml_get_name(tensor), layer, kind)) {
            int which = -1;
            if (parse_expert_aux_name(ggml_get_name(tensor), layer, kind, which)) {
                auxes.push_back({layer, kind, which, tensor});
            }
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
    // the aux rows of a routed weight (of a duplicate name, the first one)
    out.aux.assign(out.n_layers, expert_aux_tensors{});
    out.aux_nb.assign(out.n_layers, expert_aux_bytes{});
    for (const found_aux & item : auxes) {
        if (item.layer >= out.n_layers || out.tensors[item.layer][item.kind] == nullptr ||
                out.aux[item.layer][item.kind][item.which] != nullptr || !expert_aux_usable(item.tensor, experts)) {
            continue;
        }
        out.aux[item.layer][item.kind][item.which]    = item.tensor;
        out.aux_nb[item.layer][item.kind][item.which] = item.tensor->nb[1];
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

    assign_size_classes(out);
    return true;
}

inline bool build_geometry(const ggml_context * ctx, geometry & out, std::string & reason) {
    std::vector<const ggml_tensor *> tensors;
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != nullptr;
            tensor = ggml_get_next_tensor(ctx, tensor)) {
        tensors.push_back(tensor);
    }
    return build_geometry_tensors(tensors, out, reason);
}

// Joint cache: the geometries of several models laid end to end. Part i's layer l is joint layer
// offsets[i] + l, and the size classes are recomputed over the union, so two models whose expert
// slices have the same layout share a class. All parts must route the same number of experts.
inline bool concat_geometry(const std::vector<const geometry *> & parts, geometry & out,
        std::vector<int> & offsets, std::string & reason) {
    out = geometry();
    offsets.clear();
    reason.clear();
    for (const geometry * part : parts) {
        if (part == nullptr || part->n_layers <= 0) {
            reason = "a joined model has no routed layers";
            return false;
        }
        if (out.n_experts != 0 && part->n_experts != out.n_experts) {
            char message[160];
            snprintf(message, sizeof(message), "joined models route %d and %d experts; the cache needs one expert count",
                out.n_experts, part->n_experts);
            reason = message;
            return false;
        }
        out.n_experts = part->n_experts;
        offsets.push_back(out.n_layers);
        out.n_layers += part->n_layers;
        out.nb2.insert(out.nb2.end(), part->nb2.begin(), part->nb2.end());
        out.tensors.insert(out.tensors.end(), part->tensors.begin(), part->tensors.end());
        std::vector<expert_aux_tensors> aux    = part->aux;
        std::vector<expert_aux_bytes>   aux_nb = part->aux_nb;
        aux.resize(part->n_layers);
        aux_nb.resize(part->n_layers);
        out.aux.insert(out.aux.end(), aux.begin(), aux.end());
        out.aux_nb.insert(out.aux_nb.end(), aux_nb.begin(), aux_nb.end());
    }
    assign_size_classes(out);
    return true;
}

// 64-bit FNV-1a of a string, for the model keys of the joint cache.
inline uint64_t geometry_hash(const std::string & text) {
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    for (unsigned char c : text) {
        h = (h ^ c)*UINT64_C(0x100000001b3);
    }
    return h;
}

} // namespace ggml_cuda_expert
