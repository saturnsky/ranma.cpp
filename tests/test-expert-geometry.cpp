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
        CHECK(parse_expert_tensor_name("blk.0.ffn_gate_exps.weight", layer, kind));
        CHECK(layer == 0 && kind == 1);
        CHECK(!parse_expert_tensor_name("blk.0.ffn_gate_exps", layer, kind));
        CHECK(!parse_expert_tensor_name("blk.0.ffn_gate_exps.rot_in", layer, kind));
        CHECK(!parse_expert_tensor_name("blk.0.ffn_up_exps.rot_out", layer, kind));
        CHECK(!parse_expert_tensor_name("blk.0.ffn_down_exps.bias", layer, kind));
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

    // The other tensors of a bank are not experts: an EXL3 bank with rot_in / rot_out (and a bias), a gpt-oss bank
    // with biases, an NVFP4 bank with per-expert scale and input_scale. The geometry holds the weights only; an EXL3
    // weight is a quantized bank like any other, and its type id makes the signature differ from a GGUF bank of
    // the same shape.
    {
        auto add_bank = [](ggml_context * ctx, int layer, ggml_type type, int64_t emb, int64_t ff, int64_t experts,
                bool rot, bool bias, bool scale) {
            const char * kinds[3] = { "ffn_up_exps", "ffn_gate_exps", "ffn_down_exps" };
            for (int k = 0; k < 3; ++k) {
                const int64_t ne0 = k == 2 ? ff : emb;
                const int64_t ne1 = k == 2 ? emb : ff;
                char name[128];
                snprintf(name, sizeof(name), "blk.%d.%s.%s", layer, kinds[k], "weight");
                ggml_set_name(ggml_new_tensor_3d(ctx, type, ne0, ne1, experts), name);
                if (rot) {
                    snprintf(name, sizeof(name), "blk.%d.%s.rot_in", layer, kinds[k]);
                    ggml_set_name(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne0, experts), name);
                    snprintf(name, sizeof(name), "blk.%d.%s.rot_out", layer, kinds[k]);
                    ggml_set_name(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne1, experts), name);
                }
                if (bias) {
                    snprintf(name, sizeof(name), "blk.%d.%s.bias", layer, kinds[k]);
                    ggml_set_name(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne1, experts), name);
                }
                if (scale) {
                    snprintf(name, sizeof(name), "blk.%d.%s.scale", layer, kinds[k]);
                    ggml_set_name(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, experts), name);
                    snprintf(name, sizeof(name), "blk.%d.%s.input_scale", layer, kinds[k]);
                    ggml_set_name(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, experts), name);
                }
            }
        };
        struct bank_case { ggml_type type; bool rot, bias, scale; };
        const bank_case cases[] = {
            { GGML_TYPE_EXL3_M3, true,  false, false },
            { GGML_TYPE_EXL3_M3, true,  true,  false },
            { GGML_TYPE_MXFP4,   false, true,  false },
            { GGML_TYPE_NVFP4,   false, false, true  },
        };
        std::string exl3_signature;
        for (const bank_case & c : cases) {
            context_holder holder;
            CHECK(holder.ctx != nullptr);
            for (int layer = 0; layer < 4; ++layer) {
                add_bank(holder.ctx, layer, c.type, 512, 256, 16, c.rot, c.bias, c.scale);
            }
            geometry geo;
            CHECK(build_geometry(holder.ctx, geo, reason));
            CHECK(reason.empty());
            CHECK(geo.n_layers == 4 && geo.n_experts == 16 && geo.class_bytes.size() == 1);
            for (int layer = 0; layer < 4; ++layer) {
                for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                    const ggml_tensor * t = geo.tensors[layer][kind];
                    CHECK(t != nullptr && t->type == c.type);
                    const std::string name = ggml_get_name(t);
                    CHECK(name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0);
                    CHECK(geo.nb2[layer][kind] == t->nb[2]);
                }
            }
            if (c.type == GGML_TYPE_EXL3_M3) {
                char line[64];
                snprintf(line, sizeof(line), "\n0,0,%d,512,256,", (int) GGML_TYPE_EXL3_M3);
                CHECK(geo.signature().find(line) != std::string::npos);
                // bias or no bias, the same bank
                CHECK(exl3_signature.empty() || geo.signature() == exl3_signature);
                exl3_signature = geo.signature();
            }
        }
        // the same shapes as a Q4_K bank: another profile key
        context_holder holder;
        CHECK(holder.ctx != nullptr);
        for (int layer = 0; layer < 4; ++layer) {
            add_bank(holder.ctx, layer, GGML_TYPE_Q4_K, 512, 256, 16, false, false, false);
        }
        geometry geo;
        CHECK(build_geometry(holder.ctx, geo, reason));
        CHECK(geo.signature() != exl3_signature);
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

    printf("PASS: parse/build/class/signature checks on qwen-like, glm-like, mixed, companion-tensor and malformed contexts\n");
    return 0;
}
