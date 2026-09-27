#include "expert-geometry.h"

#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

using namespace ggml_cuda_expert;

namespace {

struct context_holder {
    ggml_context * ctx = nullptr;
    context_holder(size_t mem_size = 64u*1024u*1024u) {
        ggml_init_params params = {};
        params.mem_size   = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc   = true;
        ctx = ggml_init(params);
    }
    ~context_holder() { if (ctx) { ggml_free(ctx); } }
    context_holder(const context_holder &) = delete;
    context_holder & operator=(const context_holder &) = delete;
};

void add_expert_tensor(ggml_context * ctx, int layer, const char * kind, ggml_type type,
        int64_t ne0, int64_t ne1, int64_t experts) {
    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, type, ne0, ne1, experts);
    char name[128];
    snprintf(name, sizeof(name), "blk.%d.%s.weight", layer, kind);
    ggml_set_name(tensor, name);
}

void add_routed_layer(ggml_context * ctx, int layer, ggml_type up_type, ggml_type down_type,
        int64_t ff, int64_t emb, int64_t experts) {
    add_expert_tensor(ctx, layer, "ffn_up_exps",   up_type,   emb, ff,  experts);
    add_expert_tensor(ctx, layer, "ffn_gate_exps", up_type,   emb, ff,  experts);
    add_expert_tensor(ctx, layer, "ffn_down_exps", down_type, ff,  emb, experts);
}

} // namespace

int main() {
    // parse_expert_tensor_name edge cases.
    {
        int layer = -1, kind = -1;
        CHECK(parse_expert_tensor_name("blk.7.ffn_up_exps.weight", layer, kind));
        CHECK(layer == 7 && kind == 0);
        CHECK(parse_expert_tensor_name("blk.0.ffn_gate_exps", layer, kind));
        CHECK(layer == 0 && kind == 1);
        CHECK(parse_expert_tensor_name("blk.123.ffn_down_exps.weight", layer, kind));
        CHECK(layer == 123 && kind == 2);
        CHECK(!parse_expert_tensor_name("blk.x.ffn_up_exps", layer, kind));
        CHECK(!parse_expert_tensor_name("blk.7.ffn_up.weight", layer, kind));
        CHECK(!parse_expert_tensor_name("ffn_up_exps", layer, kind));
        CHECK(!parse_expert_tensor_name("blk.-1.ffn_up_exps", layer, kind));
        CHECK(!parse_expert_tensor_name(nullptr, layer, kind));
        CHECK(!parse_expert_tensor_name("token_embd.weight", layer, kind));
    }

    std::string reason;

    // Qwen-like: 48 uniform routed layers, 512 experts. up/gate Q4_K, down
    // Q5_0. Classes are per layer, so a uniform stack is a single class even
    // though the three kinds differ in type and size.
    std::string qwen_signature;
    {
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        for (int layer = 0; layer < 48; ++layer) {
            add_routed_layer(holder.ctx, layer, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, 768, 2048, 512);
        }
        geometry geo;
        CHECK(build_geometry(holder.ctx, geo, reason));
        CHECK(reason.empty());
        CHECK(geo.n_layers == 48 && geo.n_experts == 512);
        CHECK(geo.n_counts() == 48u*512u);
        CHECK(geo.n_routed_layers() == 48);
        CHECK(geo.class_bytes.size() == 1 && geo.class_layers.size() == 1);
        CHECK(geo.class_layers[0] == 48);
        for (int layer = 0; layer < 48; ++layer) {
            CHECK(geo.layer_class[layer] == 0);
            CHECK(geo.nb2[layer] == geo.class_bytes[0]);
            for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                CHECK(geo.tensors[layer][kind] != nullptr);
                CHECK(geo.nb2[layer][kind] > 0);
            }
        }
        CHECK(geo.class_total_bytes(0) ==
            geo.class_bytes[0][0] + geo.class_bytes[0][1] + geo.class_bytes[0][2]);
        qwen_signature = geo.signature();
        CHECK(qwen_signature.compare(0, 25, "ranma-expert-geometry-v1\n") == 0);

        // Same input twice must produce the same string.
        geometry again;
        CHECK(build_geometry(holder.ctx, again, reason));
        CHECK(again.signature() == qwen_signature);

        // A different bytes-per-expert stride must produce a different string.
        geometry altered = geo;
        altered.nb2[3][1] += 256;
        CHECK(altered.signature() != qwen_signature);
    }

    // GLM-like: three leading dense layers, routed layers 3..9.
    {
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        for (int layer = 0; layer < 3; ++layer) {
            char name[128];
            ggml_tensor * dense = ggml_new_tensor_2d(holder.ctx, GGML_TYPE_Q4_K, 2048, 5504);
            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", layer);
            ggml_set_name(dense, name);
        }
        for (int layer = 3; layer < 10; ++layer) {
            add_routed_layer(holder.ctx, layer, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 1536, 2048, 128);
        }
        geometry geo;
        CHECK(build_geometry(holder.ctx, geo, reason));
        CHECK(geo.n_layers == 10 && geo.n_experts == 128);
        CHECK(geo.n_routed_layers() == 7);
        for (int layer = 0; layer < 3; ++layer) {
            CHECK(geo.layer_class[layer] == -1);
            CHECK(geo.nb2[layer][0] == 0 && geo.tensors[layer][0] == nullptr);
        }
        for (int layer = 3; layer < 10; ++layer) {
            CHECK(geo.layer_class[layer] == 0);
        }
        CHECK(geo.class_layers.size() == 1 && geo.class_layers[0] == 7);
        // Dense layers contribute no signature lines.
        CHECK(geo.signature().find("\n0,0,") == std::string::npos);
    }

    // Mixed classes: the down type changes halfway through the stack.
    {
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        for (int layer = 0; layer < 4; ++layer) {
            add_routed_layer(holder.ctx, layer, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, 768, 1024, 64);
        }
        for (int layer = 4; layer < 8; ++layer) {
            add_routed_layer(holder.ctx, layer, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 768, 1024, 64);
        }
        geometry geo;
        CHECK(build_geometry(holder.ctx, geo, reason));
        CHECK(geo.class_bytes.size() == 2);
        CHECK(geo.class_layers[0] == 4 && geo.class_layers[1] == 4);
        for (int layer = 0; layer < 4; ++layer) { CHECK(geo.layer_class[layer] == 0); }
        for (int layer = 4; layer < 8; ++layer) { CHECK(geo.layer_class[layer] == 1); }
        CHECK(geo.class_total_bytes(0) != geo.class_total_bytes(1));
    }

    // A routed layer with a different ne1 is a separate class even at the same type.
    {
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        add_routed_layer(holder.ctx, 0, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 768,  1024, 32);
        add_routed_layer(holder.ctx, 1, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 1536, 1024, 32);
        add_routed_layer(holder.ctx, 2, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 768,  1024, 32);
        geometry geo;
        CHECK(build_geometry(holder.ctx, geo, reason));
        CHECK(geo.class_bytes.size() == 2);
        CHECK(geo.layer_class[0] == 0 && geo.layer_class[1] == 1 && geo.layer_class[2] == 0);
        CHECK(geo.class_layers[0] == 2 && geo.class_layers[1] == 1);
    }

    // Missing kind.
    {
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        add_routed_layer(holder.ctx, 0, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 768, 1024, 32);
        add_expert_tensor(holder.ctx, 1, "ffn_up_exps",   GGML_TYPE_Q4_K, 1024, 768, 32);
        add_expert_tensor(holder.ctx, 1, "ffn_gate_exps", GGML_TYPE_Q4_K, 1024, 768, 32);
        geometry geo;
        CHECK(!build_geometry(holder.ctx, geo, reason));
        CHECK(!reason.empty());
    }

    // Mismatched expert count.
    {
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        add_routed_layer(holder.ctx, 0, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 768, 1024, 32);
        add_routed_layer(holder.ctx, 1, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 768, 1024, 64);
        geometry geo;
        CHECK(!build_geometry(holder.ctx, geo, reason));
        CHECK(!reason.empty());
    }

    // Non-quantized expert tensor.
    {
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        add_routed_layer(holder.ctx, 0, GGML_TYPE_F16, GGML_TYPE_F16, 768, 1024, 32);
        geometry geo;
        CHECK(!build_geometry(holder.ctx, geo, reason));
        CHECK(!reason.empty());
    }

    // A view of an expert tensor is rejected.
    {
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        ggml_tensor * base = ggml_new_tensor_3d(holder.ctx, GGML_TYPE_Q4_K, 1024, 768, 32);
        ggml_set_name(base, "base.weight");
        ggml_tensor * view = ggml_view_3d(holder.ctx, base, 1024, 768, 32,
            base->nb[1], base->nb[2], 0);
        ggml_set_name(view, "blk.0.ffn_up_exps.weight");
        add_expert_tensor(holder.ctx, 0, "ffn_gate_exps", GGML_TYPE_Q4_K, 1024, 768, 32);
        add_expert_tensor(holder.ctx, 0, "ffn_down_exps", GGML_TYPE_Q4_K, 768, 1024, 32);
        geometry geo;
        CHECK(!build_geometry(holder.ctx, geo, reason));
        CHECK(!reason.empty());
    }

    // No MoE at all is a success with an empty geometry.
    {
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        ggml_tensor * embd = ggml_new_tensor_2d(holder.ctx, GGML_TYPE_Q4_K, 2048, 4096);
        ggml_set_name(embd, "token_embd.weight");
        ggml_tensor * out = ggml_new_tensor_2d(holder.ctx, GGML_TYPE_Q4_K, 2048, 4096);
        ggml_set_name(out, "output.weight");
        geometry geo;
        CHECK(build_geometry(holder.ctx, geo, reason));
        CHECK(reason.empty());
        CHECK(geo.n_layers == 0 && geo.n_experts == 0);
        CHECK(geo.n_counts() == 0);
        CHECK(geo.class_bytes.empty());
    }

    printf("PASS: parse/build/class/signature checks on qwen-like, glm-like, mixed and malformed contexts\n");
    return 0;
}
