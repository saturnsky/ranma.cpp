#include "llama-layer-exec.h"

#include "llama-batch.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-graph-chunk.h"
#include "llama-impl.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-prefetch.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

// the scratch of the expert-major MoE op (plain C++, shared with the op and the layer mode)
#include "../ggml/src/ggml-cuda/moe-ffn-had-layout.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>

static llama_layer_mode_config g_layer_mode_config;

llama_layer_mode_config llama_layer_mode_get_config() {
    return g_layer_mode_config;
}

void llama_layer_mode_set_config(const llama_layer_mode_config & cfg) {
    g_layer_mode_config = cfg;
}

// read at every batch (a test switches it between contexts)
bool llama_layer_exec_enabled() {
    return g_layer_mode_config.layer_exec;
}

int llama_layer_exec_log() {
    static const int v = [] {
        const char * env = getenv("RANMA_LAYER_EXEC_LOG");
        return env != nullptr ? atoi(env) : 0;
    }();
    return v;
}

//
// the cut
//

namespace {

enum lx_kind {
    LX_NODE,   // a node of the graph
    LX_WEIGHT, // a model weight (or an empty descriptor)
    LX_INPUT,  // a graph input
    LX_STATE,  // anything else with its own buffer: caches, recurrent and compressor states
    LX_SCRATCH, // a leaf without data the graph allocates (e.g. the shape a fill writes): every copy gets its own
    LX_BAD,
};

bool lx_is_view(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

struct lx_graph {
    ggml_cgraph * gf;
    int n;
    std::unordered_map<const ggml_tensor *, int> idx;

    explicit lx_graph(ggml_cgraph * gf) : gf(gf), n(ggml_graph_n_nodes(gf)) {
        idx.reserve((size_t) n*2);
        for (int i = 0; i < n; ++i) {
            idx[ggml_graph_node(gf, i)] = i;
        }
    }

    ggml_tensor * node(int i) const { return ggml_graph_node(gf, i); }

    int index(const ggml_tensor * t) const {
        const auto it = idx.find(t);
        return it == idx.end() ? -1 : it->second;
    }

    // the srcs of node i and its view source, each once (not the node itself: ggml_cast names its result as src 1)
    template <typename F>
    void deps(int i, F && f) const {
        const ggml_tensor * t = node(i);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (t->src[j] != nullptr && t->src[j] != t) {
                f(t->src[j]);
            }
        }
        if (t->view_src != nullptr) {
            bool seen = false;
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                seen = seen || t->src[j] == t->view_src;
            }
            if (!seen) {
                f(t->view_src);
            }
        }
    }

    int kind(const ggml_tensor * t) const {
        if (index(t) >= 0) {
            return LX_NODE;
        }
        const ggml_tensor * b = t->view_src != nullptr ? t->view_src : t;
        if ((t->flags & GGML_TENSOR_FLAG_INPUT) || (b->flags & GGML_TENSOR_FLAG_INPUT)) {
            return LX_INPUT;
        }
        if (b->buffer == nullptr) {
            return ggml_nelements(t) == 0 ? LX_WEIGHT : t->op == GGML_OP_NONE && t->view_src == nullptr ? LX_SCRATCH : LX_BAD;
        }
        return ggml_backend_buffer_get_usage(b->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS ? LX_WEIGHT : LX_STATE;
    }
};

ggml_tensor * lx_named(ggml_cgraph * gf, const char * name, int il) {
    char buf[GGML_MAX_NAME];
    snprintf(buf, sizeof(buf), "%s-%d", name, il);
    return ggml_graph_get_tensor(gf, buf);
}

// the common part of the cut: pure nodes, the router inputs and MoE outputs, the regions
struct lx_cut {
    const lx_graph & g;
    int L;

    std::vector<char> pure;     // reads only weights and inputs (directly or through pure nodes)
    std::vector<int>  region;   // the layer whose MoE region holds the node, -1 for none
    std::vector<int>  xi, yi;   // per layer: the node index of the router input and the MoE output
    std::vector<std::vector<int>> cons;

    lx_cut(const lx_graph & g, int L) : g(g), L(L) {}

    bool run(const llama_lx_names & names, std::string & err) {
        const int n = g.n;

        xi.assign(L, -1);
        yi.assign(L, -1);
        std::vector<int> x_of(n, -1);
        for (int l = 0; l < L; ++l) {
            xi[l] = g.index(lx_named(g.gf, names.moe_in,  l));
            yi[l] = g.index(lx_named(g.gf, names.moe_out, l));
            if (xi[l] < 0 || yi[l] < 0) {
                err = "no " + std::string(xi[l] < 0 ? names.moe_in : names.moe_out) + " node in layer " + std::to_string(l);
                return false;
            }
            x_of[xi[l]] = l;
        }

        pure.assign(n, 1);
        cons.assign(n, {});
        std::vector<int> dx(n, -1); // the last layer whose router input the node descends from
        bool bad = false;
        for (int i = 0; i < n; ++i) {
            g.deps(i, [&](const ggml_tensor * s) {
                const int k = g.kind(s);
                if (k == LX_NODE) {
                    const int j = g.index(s);
                    pure[i] = pure[i] && pure[j];
                    dx[i]   = std::max(dx[i], dx[j]);
                    cons[j].push_back(i);
                } else if (k == LX_STATE) {
                    pure[i] = 0;
                } else if (k == LX_BAD) {
                    bad = true;
                    err = std::string("a leaf without data: ") + s->name;
                }
            });
            if (x_of[i] >= 0) {
                dx[i] = std::max(dx[i], x_of[i]);
            }
        }
        if (bad) {
            return false;
        }

        // the region of layer l: what the MoE output reads, back to the router input; the pure nodes on the way are
        // copied where needed and are not part of it
        region.assign(n, -1);
        for (int l = 0; l < L; ++l) {
            std::vector<int> stack = { yi[l] };
            while (!stack.empty()) {
                const int v = stack.back();
                stack.pop_back();
                if (v == xi[l] || pure[v] || region[v] == l) {
                    continue;
                }
                if (region[v] >= 0 || dx[v] != l) {
                    err = std::string("the MoE of layer ") + std::to_string(l) + " reads " + g.node(v)->name +
                        " besides its router input";
                    return false;
                }
                region[v] = l;
                g.deps(v, [&](const ggml_tensor * s) {
                    const int j = g.index(s);
                    if (j >= 0) {
                        stack.push_back(j);
                    }
                });
            }
            if (region[yi[l]] != l) {
                err = "the MoE output of layer " + std::to_string(l) + " is not in its region";
                return false;
            }
        }

        // only the MoE output leaves a region
        for (int i = 0; i < n; ++i) {
            if (region[i] >= 0) {
                continue;
            }
            bool leak = false;
            g.deps(i, [&](const ggml_tensor * s) {
                const int j = g.index(s);
                if (j >= 0 && region[j] >= 0 && j != yi[region[j]]) {
                    leak = true;
                }
            });
            if (leak) {
                err = std::string(g.node(i)->name) + " reads an intermediate of a MoE region";
                return false;
            }
        }

        return true;
    }

    // the state base a dep reads, or null
    const ggml_tensor * state_base(const ggml_tensor * s) const {
        const int k = g.kind(s);
        if (k == LX_STATE) {
            return s->view_src != nullptr ? s->view_src : s;
        }
        if (k == LX_NODE && s->view_src != nullptr && lx_is_view(s->op) && g.kind(s->view_src) == LX_STATE) {
            return s->view_src;
        }
        return nullptr;
    }
};

bool lx_rows_ok(const ggml_tensor * t, int64_t n_tokens, size_t & pt) {
    int outer = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (t->ne[d] != 1) {
            outer = d;
        }
    }
    if (n_tokens > 1 && t->ne[outer] != n_tokens) {
        return false;
    }
    if (ggml_nbytes(t) % (size_t) n_tokens != 0) {
        return false;
    }
    pt = ggml_nbytes(t)/(size_t) n_tokens;
    return true;
}

bool lx_edge_ok(const ggml_tensor * t, int64_t n_tokens, size_t & pt) {
    if (t->view_src != nullptr || lx_is_view(t->op) || !ggml_is_contiguous(t) || n_tokens <= 0) {
        return false;
    }
    return lx_rows_ok(t, n_tokens, pt);
}

// an edge in the buffer of the node it is computed into: itself, or the node an in-place op writes into when the
// result covers all of it
bool lx_edge_place(const lx_graph & g, llama_lx_edge & e, int64_t n_tokens) {
    const ggml_tensor * t = e.t;
    if (t->view_src != nullptr && !lx_is_view(t->op) && g.index(t->view_src) >= 0) {
        ggml_tensor * b = t->view_src;
        // the rows are those of the result (the node may be laid out flat, e.g. a cast arange)
        if (t->view_offs != 0 || ggml_nbytes(t) != ggml_nbytes(b) || !ggml_is_contiguous(t) || b->view_src != nullptr ||
                lx_is_view(b->op) || !ggml_is_contiguous(b) || n_tokens <= 0 || !lx_rows_ok(t, n_tokens, e.pt)) {
            return false;
        }
        e.place_as = b;
        return true;
    }
    return lx_edge_ok(t, n_tokens, e.pt);
}

} // namespace

bool llama_lx_analyze(ggml_cgraph * gf, int n_layer, const llama_lx_names & names, int64_t n_tokens, llama_lx_plan & plan) {
    plan = llama_lx_plan();
    plan.n_layer  = n_layer;
    plan.n_tokens = n_tokens;

    const lx_graph g(gf);
    const int L = n_layer;
    const int n = g.n;

    lx_cut cut(g, L);
    if (!cut.run(names, plan.err)) {
        return false;
    }
    const auto & pure   = cut.pure;
    const auto & region = cut.region;

    for (int l = 0; l < L; ++l) {
        plan.x.push_back(g.node(cut.xi[l]));
        plan.y.push_back(g.node(cut.yi[l]));
    }

    auto fail = [&](const std::string & msg) {
        plan.err = msg;
        return false;
    };

    // the earliest segment of every node: after the MoE outputs it reads
    std::vector<int> lo(n, 0);
    for (int i = 0; i < n; ++i) {
        if (region[i] >= 0 || pure[i]) {
            continue;
        }
        g.deps(i, [&](const ggml_tensor * s) {
            const int j = g.index(s);
            if (j < 0) {
                return;
            }
            const int v = region[j] >= 0 ? region[j] + 1 : pure[j] ? 0 : lo[j];
            lo[i] = std::max(lo[i], v);
        });
    }

    // the writes into a state, by their earliest segment; the reads of that state run in the same segment
    std::map<const ggml_tensor *, int> wseg;
    std::vector<char> writer(n, 0);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = g.node(i);
        if (region[i] >= 0 || pure[i] || lx_is_view(t->op) || t->view_src == nullptr || g.kind(t->view_src) != LX_STATE) {
            continue;
        }
        writer[i] = 1;
        const auto it = wseg.find(t->view_src);
        if (it != wseg.end() && it->second != lo[i]) {
            return fail(std::string("the state ") + t->view_src->name + " is written in two segments");
        }
        wseg[t->view_src] = lo[i];
    }

    // the segment: as late as the consumers allow (a view, or a node of no state), where the state it reads is
    // written, or as early as possible (a write)
    const int INF = INT_MAX;
    std::vector<int> seg(n, -1);
    std::vector<char> fixed(n, 0);  // in the segment of the state it reads or writes
    std::vector<int>  smin(n, 0);   // the earliest segment the appendable states it reads allow
    for (int i = n - 1; i >= 0; --i) {
        if (region[i] >= 0 || pure[i]) {
            continue;
        }
        const ggml_tensor * t = g.node(i);
        int hi = INF;
        for (int c : cut.cons[i]) {
            hi = std::min(hi, region[c] >= 0 ? region[c] : seg[c]);
        }
        int s = -1;
        if (lx_is_view(t->op)) {
            s = hi == INF ? lo[i] : hi;
        } else {
            int sr = -1;
            int s_min = 0; // an appendable state is read in the segment that writes it or later
            bool conflict = false;
            g.deps(i, [&](const ggml_tensor * d) {
                const ggml_tensor * b = cut.state_base(d);
                if (b == nullptr) {
                    return;
                }
                const auto it = wseg.find(b);
                if (it == wseg.end()) {
                    return;
                }
                if (names.appendable != nullptr && names.appendable->count(b) > 0) {
                    s_min = std::max(s_min, it->second);
                    return;
                }
                conflict = conflict || (sr >= 0 && sr != it->second);
                sr = it->second;
            });
            if (conflict) {
                return fail(std::string(t->name) + " reads states written in different segments");
            }
            s = sr >= 0 ? sr : writer[i] ? lo[i] : hi == INF ? lo[i] : hi;
            fixed[i] = sr >= 0 || writer[i];
            smin[i]  = s_min;
            if (sr < 0 && s < s_min) {
                s = s_min;
            }
            if (s < s_min) {
                return fail(std::string(t->name) + " reads a state before the segment that writes it");
            }
        }
        if (s < lo[i] || (hi != INF && s > hi) || s > L) {
            return fail(std::string(t->name) + " has no segment (earliest " + std::to_string(lo[i]) + ", latest " +
                (hi == INF ? std::string("none") : std::to_string(hi)) + ", chosen " + std::to_string(s) + ")");
        }
        seg[i] = s;
    }
    // a node placed after a tensor that cannot cross segments (not a tensor of token rows, e.g. the top-k of one attention
    // chunk that a chunk join writes into the top-k of the ubatch) moves to the segment of that tensor; its readers follow
    // where it cannot cross either. Through views: a view is made again in every segment that reads it.
    {
        auto root_of = [&](int j) {
            while (j >= 0 && lx_is_view(g.node(j)->op)) {
                j = g.node(j)->src[0] != nullptr ? g.index(g.node(j)->src[0]) : -1;
            }
            return j;
        };
        std::vector<char> crosses(n, 2); // 2: not checked yet
        auto can_cross = [&](int j) {
            if (crosses[j] == 2) {
                llama_lx_edge e;
                e.t = g.node(j);
                crosses[j] = lx_edge_place(g, e, n_tokens) ? 1 : 0;
            }
            return crosses[j] == 1;
        };
        for (bool changed = true; changed; ) {
            changed = false;
            for (int i = 0; i < n; ++i) {
                if (region[i] >= 0 || pure[i] || lx_is_view(g.node(i)->op)) {
                    continue;
                }
                std::string err;
                std::vector<int> stack = { i };
                std::set<int> seen;
                while (!stack.empty() && err.empty()) {
                    const int v = stack.back();
                    stack.pop_back();
                    g.deps(v, [&](const ggml_tensor * d) {
                        const int j0 = g.index(d);
                        if (j0 < 0 || !seen.insert(j0).second) {
                            return;
                        }
                        if (lx_is_view(g.node(j0)->op)) {
                            stack.push_back(j0);
                        }
                        const int j = root_of(j0);
                        if (j < 0 || region[j] >= 0 || pure[j]) {
                            return;
                        }
                        if (seg[j] > seg[i]) {
                            // a node moved before a tensor it reads: that tensor moves along
                            if (fixed[j] || lo[j] > seg[i] || smin[j] > seg[i]) {
                                err = std::string(g.node(i)->name) + " (segment " + std::to_string(seg[i]) + ") reads " +
                                    g.node(j)->name + " of the later segment " + std::to_string(seg[j]);
                                return;
                            }
                            seg[j] = seg[i];
                            changed = true;
                            return;
                        }
                        if (seg[j] == seg[i] || can_cross(j)) {
                            return;
                        }
                        if (fixed[i] || lo[i] > seg[j] || smin[i] > seg[j]) {
                            err = std::string(g.node(i)->name) + " (segment " + std::to_string(seg[i]) + ") reads " + g.node(j)->name +
                                " of segment " + std::to_string(seg[j]) + ", which cannot cross segments";
                            return;
                        }
                        seg[i] = seg[j];
                        changed = true;
                    });
                }
                if (!err.empty()) {
                    return fail(err);
                }
            }
        }
    }
    for (int l = 0; l < L; ++l) {
        if (seg[cut.xi[l]] != l) {
            return fail("the router input of layer " + std::to_string(l) + " is in segment " + std::to_string(seg[cut.xi[l]]));
        }
    }

    // the nodes of every segment and the tensors that cross from one segment to a later one
    plan.nodes.assign(L + 1, {});
    plan.out.assign(L + 1, {});
    std::vector<std::set<int>> out_set(L + 1);
    std::vector<int> last_read(n, -1); // per crossing tensor: the last segment that reads it
    for (int l = 0; l < L; ++l) {
        out_set[l].insert(cut.xi[l]);
    }
    std::set<const ggml_tensor *> inputs;
    plan.seg_input_bytes.assign(L + 1, 0);
    plan.seg_dev_masks.assign(L + 1, 0);
    plan.seg_host_mask_bytes.assign(L + 1, 0);
    auto is_host_mask = [](const ggml_tensor * t) {
        const size_t len = strlen(t->name);
        return len >= 7 && strcmp(t->name + len - 7, "kq_mask") == 0;
    };
    std::set<const ggml_tensor *> host_masks;
    for (int s = 0; s <= L; ++s) {
        std::vector<char> need(n, 0);
        std::vector<int> stack;
        std::set<const ggml_tensor *> seg_inputs;
        for (int i = 0; i < n; ++i) {
            // a view another node reads comes with its readers (each segment makes its own)
            if (seg[i] == s && !(lx_is_view(g.node(i)->op) && !cut.cons[i].empty())) {
                need[i] = 1;
                stack.push_back(i);
            }
        }
        std::string err;
        while (!stack.empty() && err.empty()) {
            const int i = stack.back();
            stack.pop_back();
            g.deps(i, [&](const ggml_tensor * d) {
                const int j = g.index(d);
                if (j < 0) {
                    if (g.kind(d) == LX_INPUT) {
                        const ggml_tensor * in = d->view_src != nullptr && (d->view_src->flags & GGML_TENSOR_FLAG_INPUT) ? d->view_src : d;
                        inputs.insert(in);
                        seg_inputs.insert(in);
                    }
                    return;
                }
                if (need[j]) {
                    return;
                }
                if (region[j] >= 0) {
                    if (j != cut.yi[region[j]] || region[j] != s - 1) {
                        err = std::string(g.node(i)->name) + " reads the MoE output of another layer";
                    }
                    return;
                }
                // a pure node, or a view of a tensor of an earlier segment (each segment that reads it makes its own)
                if (pure[j] || lx_is_view(g.node(j)->op)) {
                    need[j] = 1;
                    stack.push_back(j);
                    return;
                }
                if (seg[j] >= 0 && seg[j] < s) {
                    last_read[j] = std::max(last_read[j], s);
                    return;
                }
                err = std::string(g.node(i)->name) + " (segment " + std::to_string(s) + ") reads " + d->name + " of segment " +
                    std::to_string(seg[j]);
            });
        }
        if (!err.empty()) {
            return fail(err);
        }
        for (const ggml_tensor * t : seg_inputs) {
            plan.seg_input_bytes[s] += ggml_nbytes(t);
            if (is_host_mask(t)) {
                plan.seg_host_mask_bytes[s] += ggml_nbytes(t);
                host_masks.insert(t);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (need[i]) {
                plan.nodes[s].push_back(i);
                plan.seg_dev_masks[s] += strncmp(g.node(i)->name, "chunk_mask", 10) == 0;
            }
        }
    }
    // an in-place result and the node it writes into are one tensor: the result carries it for both
    for (int j = 0; j < n; ++j) {
        if (last_read[j] < 0) {
            continue;
        }
        llama_lx_edge e;
        e.t = g.node(j);
        if (lx_edge_place(g, e, n_tokens) && e.place_as != nullptr) {
            const int r = g.index(e.place_as);
            if (r >= 0 && last_read[r] >= 0) {
                if (seg[r] != seg[j]) {
                    return fail(std::string(e.t->name) + " writes in place into a tensor of another segment");
                }
                last_read[j] = std::max(last_read[j], last_read[r]);
                last_read[r] = -1;
            }
        }
    }

    // a tensor read by the next segment only is an output of its segment; one read later is carried in a buffer of its own
    std::set<int> carry_set;
    for (int j = 0; j < n; ++j) {
        if (last_read[j] < 0) {
            continue;
        }
        if (last_read[j] == seg[j] + 1) {
            out_set[seg[j]].insert(j);
        } else {
            carry_set.insert(j);
        }
    }
    for (int l = 0; l < L; ++l) {
        if (carry_set.count(cut.xi[l]) > 0) {
            return fail("the router input of layer " + std::to_string(l) + " is read past the next segment");
        }
    }
    for (int s = 0; s <= L; ++s) {
        // the router input first: slot 0 holds the router input of every layer, its rows contiguous for the MoE
        std::vector<int> order(out_set[s].begin(), out_set[s].end());
        if (s < L) {
            order.erase(std::find(order.begin(), order.end(), cut.xi[s]));
            order.insert(order.begin(), cut.xi[s]);
        }
        for (int j : order) {
            llama_lx_edge e;
            e.node   = j;
            e.t      = g.node(j);
            e.s      = s;
            e.t_last = s + 1;
            if (!lx_edge_place(g, e, n_tokens)) {
                return fail(std::string("the segment output ") + e.t->name + " (" + ggml_op_desc(e.t) + ", view of " +
                    (e.t->view_src != nullptr ? e.t->view_src->name : "nothing") + ") is not a contiguous tensor of token rows");
            }
            plan.out[s].push_back(e);
        }
    }
    for (int j : carry_set) {
        llama_lx_edge e;
        e.node   = j;
        e.t      = g.node(j);
        e.s      = seg[j];
        e.t_last = last_read[j];
        if (!lx_edge_place(g, e, n_tokens)) {
            return fail(std::string("the carried tensor ") + e.t->name + " (" + ggml_op_desc(e.t) + ", view of " +
                (e.t->view_src != nullptr ? e.t->view_src->name : "nothing") + ") is not a contiguous tensor of token rows");
        }
        plan.carry.push_back(e);
    }
    for (const ggml_tensor * t : inputs) {
        plan.input_bytes += ggml_nbytes(t);
    }
    for (const ggml_tensor * t : host_masks) {
        plan.host_mask_bytes += ggml_nbytes(t);
    }
    {
        std::set<const ggml_tensor *> all;
        for (int i = 0; i < n; ++i) {
            g.deps(i, [&](const ggml_tensor * d) {
                if (g.index(d) < 0 && g.kind(d) == LX_INPUT) {
                    all.insert(d->view_src != nullptr && (d->view_src->flags & GGML_TENSOR_FLAG_INPUT) ? d->view_src : d);
                }
            });
        }
        plan.inputs.assign(all.begin(), all.end());
    }

    // one buffer set for all layers: in every segment, each node that reads an input of the segment (an output of the
    // previous one or the MoE output) is an ancestor of each output the segment writes, so the writes come later in any
    // order the backend may choose
    for (int s = 1; s < L && plan.one_set; ++s) {
        std::set<int> in;
        for (const auto & e : plan.out[s - 1]) {
            in.insert(e.node);
        }
        in.insert(cut.yi[s - 1]);
        std::vector<int> readers;
        for (int i : plan.nodes[s]) {
            bool r = false;
            g.deps(i, [&](const ggml_tensor * d) {
                const int j = g.index(d);
                r = r || (j >= 0 && in.count(j) > 0);
            });
            if (r) {
                readers.push_back(i);
            }
        }
        if (readers.size() > 64) {
            plan.one_set = false;
            break;
        }
        std::unordered_map<int, uint64_t> bits;
        for (int i : plan.nodes[s]) {
            uint64_t b = 0;
            g.deps(i, [&](const ggml_tensor * d) {
                const auto it = bits.find(g.index(d));
                if (it != bits.end()) {
                    b |= it->second;
                }
            });
            const auto r = std::find(readers.begin(), readers.end(), i);
            if (r != readers.end()) {
                b |= 1ull << (r - readers.begin());
            }
            bits[i] = b;
        }
        const uint64_t all = readers.size() == 64 ? ~0ull : (1ull << readers.size()) - 1;
        for (const auto & e : plan.out[s]) {
            // an in-place result writes from the node it writes into on
            const int w = e.place_as != nullptr ? g.index(e.place_as) : e.node;
            if (w < 0 || (bits[w] & all) != all || (bits[e.node] & all) != all) {
                plan.one_set = false;
            }
        }
    }

    return true;
}

bool llama_lx_regions(ggml_cgraph * gf, int n_layer, const llama_lx_names & names, std::vector<llama_lx_region> & out,
                      std::vector<ggml_tensor *> & inputs, std::string & err) {
    const lx_graph g(gf);
    lx_cut cut(g, n_layer);
    if (!cut.run(names, err)) {
        return false;
    }
    out.assign(n_layer, {});
    std::set<ggml_tensor *> inp;
    for (int l = 0; l < n_layer; ++l) {
        auto & r = out[l];
        r.x = g.node(cut.xi[l]);
        r.y = g.node(cut.yi[l]);
        const int64_t n_tokens = r.x->ne[1];
        size_t pt = 0;
        if (!lx_edge_ok(r.x, n_tokens, pt) || !lx_edge_ok(r.y, n_tokens, pt)) {
            err = "the router input or the MoE output of layer " + std::to_string(l) + " is not a contiguous tensor of token rows";
            return false;
        }
        std::vector<char> need(g.n, 0);
        std::vector<int> stack;
        for (int i = 0; i < g.n; ++i) {
            if (cut.region[i] == l) {
                need[i] = 1;
                stack.push_back(i);
            }
        }
        while (!stack.empty()) {
            const int i = stack.back();
            stack.pop_back();
            g.deps(i, [&](ggml_tensor * d) {
                const int j = g.index(d);
                if (j < 0) {
                    if (g.kind(d) == LX_INPUT) {
                        inp.insert(d->view_src != nullptr && (d->view_src->flags & GGML_TENSOR_FLAG_INPUT) ? d->view_src : d);
                    }
                    return;
                }
                if (!need[j] && cut.pure[j]) {
                    need[j] = 1;
                    stack.push_back(j);
                }
            });
        }
        for (int i = 0; i < g.n; ++i) {
            if (need[i]) {
                r.nodes.push_back(i);
            }
        }
    }
    inputs.assign(inp.begin(), inp.end());
    return true;
}

//
// the run
//

namespace {

// a copy of some nodes of a graph, with the tensors of other segments replaced by leaves in owned buffers
struct lx_copy {
    ggml_context_ptr ctx;
    ggml_cgraph * gf = nullptr;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> map;

    bool init(size_t n_tensors, size_t graph_size) {
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(2*n_tensors + 64) + ggml_graph_overhead_custom(graph_size, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(ip));
        if (!ctx) {
            return false;
        }
        gf = ggml_new_graph_custom(ctx.get(), graph_size, false);
        return gf != nullptr;
    }

    // a leaf of the shape of t at addr in buf (also standing for the node an in-place t wrote into)
    ggml_tensor * leaf(const ggml_tensor * t, ggml_backend_buffer_t buf, char * addr, const ggml_tensor * alias = nullptr) {
        ggml_tensor * d = ggml_new_tensor(ctx.get(), t->type, GGML_MAX_DIMS, t->ne);
        ggml_format_name(d, "lx-%s", t->name);
        if (ggml_backend_tensor_alloc(buf, d, addr) != GGML_STATUS_SUCCESS) {
            return nullptr;
        }
        map[t] = d;
        if (alias != nullptr) {
            map[alias] = d;
        }
        return d;
    }

    // copy the nodes (indices into src, in graph order) and add them to the graph; the nodes in `place` are computed
    // into the given addresses
    bool copy(const lx_graph & src, const std::vector<int> & nodes,
              const std::map<const ggml_tensor *, std::pair<ggml_backend_buffer_t, char *>> & place, std::string & err) {
        for (int i : nodes) {
            const ggml_tensor * t = src.node(i);
            if (map.count(t) > 0) {
                continue; // a leaf holds it (the node an in-place input wrote into)
            }
            ggml_tensor * d = ggml_new_tensor(ctx.get(), t->type, GGML_MAX_DIMS, t->ne);
            for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                d->nb[k] = t->nb[k];
            }
            d->op    = t->op;
            d->flags = t->flags & ~GGML_TENSOR_FLAG_COMPUTE;
            memcpy(d->op_params, t->op_params, sizeof(t->op_params));
            ggml_set_name(d, t->name);

            auto get = [&](ggml_tensor * s) -> ggml_tensor * {
                if (s == t) {
                    return d; // ggml_cast
                }
                const auto it = map.find(s);
                if (it != map.end()) {
                    return it->second;
                }
                if (src.index(s) >= 0) {
                    err = std::string("the copy of ") + t->name + " (node " + std::to_string(i) + ", " + ggml_op_desc(t) + ") reads " +
                        s->name + " (node " + std::to_string(src.index(s)) + ", " + ggml_op_desc(s) + "), which it does not hold";
                    return nullptr;
                }
                if (s->buffer == nullptr && s->view_src == nullptr && s->op == GGML_OP_NONE && !(s->flags & GGML_TENSOR_FLAG_INPUT) &&
                        ggml_nelements(s) > 0) {
                    // a leaf the graph allocates: its own in this copy
                    ggml_tensor * c = ggml_new_tensor(ctx.get(), s->type, GGML_MAX_DIMS, s->ne);
                    ggml_set_name(c, s->name);
                    map[s] = c;
                    return c;
                }
                return s; // a weight, an input or a state
            };
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                if (t->src[k] != nullptr) {
                    d->src[k] = get(t->src[k]);
                    if (d->src[k] == nullptr) {
                        return false;
                    }
                }
            }
            if (t->view_src != nullptr) {
                d->view_src = get(t->view_src);
                if (d->view_src == nullptr) {
                    return false;
                }
                d->view_offs = t->view_offs;
            }
            const auto p = place.find(t);
            if (p != place.end()) {
                if (ggml_backend_tensor_alloc(p->second.first, d, p->second.second) != GGML_STATUS_SUCCESS) {
                    err = std::string("cannot place ") + t->name;
                    return false;
                }
            }
            map[t] = d;
        }
        for (int i : nodes) {
            ggml_tensor * d = map[src.node(i)];
            if (d->op != GGML_OP_NONE || d->view_src != nullptr) {
                ggml_build_forward_expand(gf, d);
            }
        }
        return true;
    }
};

struct lx_chunk {
    llm_graph_result_ptr    res;
    ggml_backend_buffer_ptr inp_buf;
    int64_t i0        = 0;
    int64_t n         = 0;
    int32_t n_outputs = 0;
    llama_lx_plan plan;
    std::vector<std::pair<const ggml_tensor *, ggml_backend_t>> backends;
};

size_t lx_pad(size_t x, size_t a) {
    return (x + a - 1)/a*a;
}

} // namespace

// The backends the builder assigned to nodes of gf (MoE descriptors, layer ends), kept for the copies: the scheduler
// forgets them when it is reset. Asking enters every node into the scheduler's hash set without marking it used, so the
// scheduler is reset here through an assignment (a reset of an unused scheduler keeps its hash set).
static void lx_record_backends(ggml_backend_sched_t sched, ggml_cgraph * gf, const std::vector<int> * only,
                               std::vector<std::pair<const ggml_tensor *, ggml_backend_t>> & out, ggml_backend_t any) {
    auto one = [&](ggml_tensor * t) {
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, t);
        if (b != nullptr) {
            out.emplace_back(t, b);
        }
    };
    if (only != nullptr) {
        for (int i : *only) {
            one(ggml_graph_node(gf, i));
        }
    } else {
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            one(ggml_graph_node(gf, i));
        }
    }
    if (ggml_graph_n_nodes(gf) > 0) {
        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(gf, 0), any);
    }
    ggml_backend_sched_reset(sched);
}

// allocate the input tensors of a built graph in one host buffer: those in `only`, and those gf holds as leaves (an
// input expanded on its own, read by no node, is still allocated and set like in a scheduled graph)
static ggml_backend_buffer_t lx_alloc_inputs(ggml_context * ctx, ggml_backend_buffer_type_t buft, const std::set<const ggml_tensor *> * only,
                                             ggml_cgraph * gf = nullptr) {
    std::vector<ggml_tensor *> ts;
    size_t size = 0;
    const size_t align = ggml_backend_buft_get_alignment(buft);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (!(t->flags & GGML_TENSOR_FLAG_INPUT) || t->data != nullptr || t->view_src != nullptr) {
            continue;
        }
        if (only != nullptr && only->count(t) == 0 && (gf == nullptr || ggml_graph_get_tensor(gf, t->name) != t)) {
            continue;
        }
        ts.push_back(t);
        size += lx_pad(ggml_backend_buft_get_alloc_size(buft, t), align);
    }
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, std::max<size_t>(size, align));
    if (buf == nullptr) {
        return nullptr;
    }
    char * p = (char *) ggml_backend_buffer_get_base(buf);
    for (ggml_tensor * t : ts) {
        if (ggml_backend_tensor_alloc(buf, t, p) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            return nullptr;
        }
        p += lx_pad(ggml_backend_buft_get_alloc_size(buft, t), align);
    }
    return buf;
}

namespace {

// a graph of the reserve memory context (a MoE donor): the MoE regions of its width, read by the MoE of every layer
struct lx_donor {
    std::unique_ptr<llama_batch_allocr> ba;
    llm_graph_result_ptr res;
    std::vector<llama_lx_region> regions;
    std::vector<ggml_tensor *> inputs;
    std::vector<std::pair<const ggml_tensor *, ggml_backend_t>> backends;
    ggml_backend_buffer_ptr inp_buf;
};

// the planned buffer of a carried tensor: by its segment and its order among the carried tensors of that segment
struct lx_carry_plan {
    int    t_last = -1;
    size_t pt     = 0;
    int    slot   = -1;
};

int64_t lx_env_i64(const char * name) {
    const char * env = getenv(name);
    return env != nullptr ? atoll(env) : 0;
}

int64_t lx_cfg_i64(int64_t llama_layer_mode_config::* field) {
    return std::max<int64_t>(0, g_layer_mode_config.*field);
}

} // namespace

int llama_context::layer_exec_decode(llama_memory_context_i * mctx, uint32_t n_tokens_all, uint32_t n_outputs_all) {
    if (!llama_layer_exec_enabled() || mctx == nullptr) {
        return 1;
    }
    const int log = llama_layer_exec_log();
    // every graph of this call makes its attention masks on the device, one chunk per ubatch included: the chunk inputs
    // are all held at once, and the host masks of the compressed layers are about N^2/2 bytes for N tokens whatever the
    // ubatch size
    llm_graph_chunk_mask_one_scope one_chunk_masks;
    auto skip = [&](const char * why) {
        if (log > 0) {
            LLAMA_LOG_INFO("%s: layer-major prefill not used: %s\n", __func__, why);
        }
        return 1;
    };

    const llama_ubatch & ub0 = mctx->get_ubatch();
    if (ub0.n_tokens >= n_tokens_all) {
        return 1;
    }

    // ---- what it supports, decided before the memory changes ---------------------------------------------------------
    const int L_all = (int) model.hparams.n_layer();
    int L = L_all; // the layers run layer by layer; the layers after them run in the last segment of every ubatch
    llama_lx_names names;
    switch (model.arch) {
        case LLM_ARCH_DEEPSEEK4:
            names.moe_in = "ffn_norm"; names.moe_out = "ffn_moe_out";
            break;
        case LLM_ARCH_DEEPSEEK41:
            names.moe_in = "ffn_norm"; names.moe_out = "ffn_moe_out";
            L = llama_dsv41_layer_exec_layers(model);
            break;
        default: return skip("architecture");
    }
    if (L < 1) {
        return skip("no layer before the tail prefill");
    }
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
        return skip("not a decoder context");
    }
    if (cparams.embeddings || cparams.embeddings_nextn || !sampling.samplers.empty() || !cparams.causal_attn ||
            cparams.pipeline_parallel || (loras && !loras->empty())) {
        return skip("embeddings, samplers, non-causal attention, pipeline parallelism or LoRA");
    }
    for (size_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            return skip("layer input extraction");
        }
    }
    const llama_batch & batch = balloc->get_batch();
    if (batch.embd != nullptr || batch.token == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return skip("not a token batch");
    }
    const llama_seq_id seq = batch.seq_id[0][0];
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != seq) {
            return skip("more than one sequence");
        }
    }

    ggml_backend_dev_t dev = model.dev_layer(0);
    for (int il = 1; il < L_all; ++il) {
        if (model.dev_layer(il) != dev) {
            return skip("the layers are on more than one device");
        }
    }
    ggml_backend_t backend_dev = nullptr;
    for (const auto & b : backends) {
        if (ggml_backend_get_device(b.get()) == dev) {
            backend_dev = b.get();
        }
    }
    if (backend_dev == nullptr) {
        return skip("no backend of the layer device");
    }

    // DeepSeek-V4.1: the compressed KV and index key caches, which the layers after a source read (see llama_lx_names)
    std::set<const ggml_tensor *> appendable;
    ggml_context_ptr appendable_ctx;
    if (model.arch == LLM_ARCH_DEEPSEEK41) {
        const auto * m4 = dynamic_cast<const llama_kv_cache_dsv4_context *>(mctx);
        if (m4 == nullptr) {
            return skip("not the DeepSeek-V4 memory");
        }
        ggml_init_params ip = { ggml_tensor_overhead()*(size_t) (4*L_all + 8), nullptr, true };
        appendable_ctx.reset(ggml_init(ip));
        for (int il = 0; il < L_all; ++il) {
            if (model.hparams.dsv4_compress_ratios[il] == 0) {
                continue;
            }
            for (const llama_kv_cache_dsv4_comp_context * comp : { m4->get_csa(), m4->get_lid() }) {
                if (comp == nullptr) {
                    continue;
                }
                try {
                    const ggml_tensor * t = comp->get_k(appendable_ctx.get(), il);
                    while (t != nullptr && t->view_src != nullptr) {
                        t = t->view_src;
                    }
                    if (t != nullptr) {
                        appendable.insert(t);
                    }
                } catch (const std::exception &) {
                    // a layer without storage in this cache
                }
            }
        }
        names.appendable = &appendable;
    }

    const int64_t t_start = ggml_time_us();
    ggml_backend_sched_t sc = sched.get();
    auto trace = [&](const char * what, int64_t a = -1, int64_t b = -1) {
        if (log > 1) {
            fprintf(stderr, "layer-major trace: %s %lld %lld\n", what, (long long) a, (long long) b);
            fflush(stderr);
        }
    };

    const int64_t C = cparams.n_ubatch;
    const int64_t N = n_tokens_all;
    const int64_t K = (N + C - 1)/C;

    const auto save_n_outputs = n_outputs;
    trace("start", N, C);
    graph_reuse_invalidate();
    gf_res_prev_active = nullptr;
    for (auto & r : gf_res_prev) {
        if (r) {
            r->reset();
        }
    }

    // a graph of n tokens of the reserve memory context, for the cut before the memory changes and for the MoE of
    // a group of ubatches
    auto build_reserve = [&](int64_t n, llama_batch_allocr & ba, llama_memory_context_i * mc) -> llm_graph_result_ptr {
        llama_ubatch ub = ba.ubatch_reserve((uint32_t) n, 1);
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            ub.n_seq_id[i] = 1;
            ub.seq_id[i]   = &ub.seq_id_unq[0];
            ub.output[i]   = i + 1 == ub.n_tokens;
        }
        n_outputs = 1;
        llm_graph_result_ptr res(new llm_graph_result(graph_max_nodes((uint32_t) n)));
        const auto gparams = graph_params(res.get(), ub, mc, LLM_GRAPH_TYPE_DEFAULT);
        res->reset();
        ggml_backend_sched_reset(sc);
        if (model.build_graph(gparams) == nullptr) {
            return nullptr;
        }
        res->trim();
        return res;
    };

    auto mctx_full = memory->init_full();
    if (!mctx_full) {
        n_outputs = save_n_outputs;
        return skip("no reserve memory context");
    }

    // the device buffer of the scheduler the MoE graph of n rows needs, from the region of layer 0 of gf (its router input
    // a leaf the measure counts)
    int backend_dev_id = -1;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sc); ++i) {
        if (ggml_backend_sched_get_backend(sc, i) == backend_dev) {
            backend_dev_id = i;
        }
    }
    bool fused = false; // the MoE is the expert-major op (MOE_FFN_HAD), which takes its rows in passes of its own
    auto moe_need = [&](ggml_cgraph * gf, const llama_lx_region & r,
                        const std::vector<std::pair<const ggml_tensor *, ggml_backend_t>> & assigned) -> size_t {
        const lx_graph g(gf);
        lx_copy cp;
        if (backend_dev_id < 0 || !cp.init(r.nodes.size() + 1, r.nodes.size()*4 + 256)) {
            return SIZE_MAX;
        }
        cp.map[r.x] = ggml_new_tensor(cp.ctx.get(), r.x->type, GGML_MAX_DIMS, r.x->ne);
        std::string err;
        if (!cp.copy(g, r.nodes, {}, err)) {
            return SIZE_MAX;
        }
        for (int i : r.nodes) {
            fused = fused || g.node(i)->op == GGML_OP_MOE_FFN_HAD;
        }
        ggml_backend_sched_reset(sc);
        for (const auto & a : assigned) {
            const auto it = cp.map.find(a.first);
            if (it != cp.map.end()) {
                ggml_backend_sched_set_tensor_backend(sc, it->second, a.second);
            }
        }
        std::vector<size_t> sizes((size_t) ggml_backend_sched_get_n_backends(sc), 0);
        ggml_backend_sched_reserve_size(sc, cp.gf, sizes.data());
        ggml_backend_sched_reset(sc);
        return sizes[(size_t) backend_dev_id];
    };

    // the cut of the chunk shapes: the edges per segment and their sizes
    std::vector<size_t> slot_pt;      // per edge slot: the largest bytes per token
    std::map<std::pair<int, int>, lx_carry_plan> carry_plan; // (segment, order) -> the buffer of a carried tensor
    size_t input_bytes_max = 0;
    size_t meta_bytes_max  = 0;
    size_t moe_need_c      = 0;       // the scheduler buffer of the MoE of one ubatch (C rows)
    size_t y_pt            = 0;       // MoE output bytes per row
    bool one_set = true;
    {
        std::vector<int64_t> shapes = { C };
        if (N - (K - 1)*C != C) {
            shapes.push_back(N - (K - 1)*C);
        }
        for (int64_t n : shapes) {
            trace("reserve build", n);
            llama_batch_allocr ba(model.hparams.n_pos_per_embd());
            auto res = build_reserve(n, ba, mctx_full.get());
            trace("reserve built", res ? ggml_graph_n_nodes(res->get_gf()) : -1);
            llama_lx_plan p;
            if (!res || !llama_lx_analyze(res->get_gf(), L, names, n, p)) {
                n_outputs = save_n_outputs;
                ggml_backend_sched_reset(sc);
                if (log > 0 && res) {
                    LLAMA_LOG_INFO("%s: the graph of %" PRId64 " tokens does not cut: %s\n", __func__, n, p.err.c_str());
                }
                return skip("the graph does not cut");
            }
            for (int s = 0; s <= L; ++s) {
                for (size_t j = 0; j < p.out[s].size(); ++j) {
                    if (slot_pt.size() <= j) {
                        slot_pt.push_back(0);
                    }
                    slot_pt[j] = std::max(slot_pt[j], p.out[s][j].pt);
                }
            }
            {
                std::map<int, int> ord;
                for (const auto & e : p.carry) {
                    auto & cpl = carry_plan[{ e.s, ord[e.s]++ }];
                    cpl.t_last = std::max(cpl.t_last, e.t_last);
                    cpl.pt     = std::max(cpl.pt, e.pt);
                }
            }
            one_set = one_set && p.one_set;
            input_bytes_max = std::max(input_bytes_max, p.input_bytes);
            meta_bytes_max  = std::max(meta_bytes_max, (size_t) ggml_graph_n_nodes(res->get_gf())*ggml_tensor_overhead()*2);
            if (n == C) {
                std::vector<llama_lx_region> regs;
                std::vector<ggml_tensor *> inps;
                std::string err;
                if (!llama_lx_regions(res->get_gf(), L, names, regs, inps, err)) {
                    n_outputs = save_n_outputs;
                    ggml_backend_sched_reset(sc);
                    if (log > 0) {
                        LLAMA_LOG_INFO("%s: the MoE of a ubatch does not cut: %s\n", __func__, err.c_str());
                    }
                    return skip("the MoE regions do not cut");
                }
                std::vector<std::pair<const ggml_tensor *, ggml_backend_t>> bk;
                lx_record_backends(sc, res->get_gf(), &regs[0].nodes, bk, backend_dev);
                moe_need_c = moe_need(res->get_gf(), regs[0], bk);
                y_pt = ggml_nbytes(regs[0].y)/(size_t) n;
            }
        }
    }
    ggml_backend_sched_reset(sc);
    if (moe_need_c == SIZE_MAX || y_pt == 0) {
        n_outputs = save_n_outputs;
        return skip("the MoE of a ubatch cannot be measured");
    }

    // the buffers of the carried tensors: one per tensor while it lives, a buffer freed when its last reader ran (a
    // buffer written in segment s is free when the segment of its last read is before s)
    std::vector<std::pair<size_t, int>> carry_slots; // per slot: bytes per token, the last segment that reads it
    for (auto & kv : carry_plan) {
        const int s = kv.first.first;
        int pick = -1;
        for (size_t k = 0; k < carry_slots.size() && pick < 0; ++k) {
            if (carry_slots[k].second < s) {
                pick = (int) k;
            }
        }
        if (pick < 0) {
            carry_slots.emplace_back(0, -1);
            pick = (int) carry_slots.size() - 1;
        }
        carry_slots[(size_t) pick].first  = std::max(carry_slots[(size_t) pick].first, kv.second.pt);
        carry_slots[(size_t) pick].second = kv.second.t_last;
        kv.second.slot = pick;
    }

    // ---- the plan: ubatches per group, MoE bundles --------------------------------------------------------------------
    // Per token row of a group: the segment edges (one or two sets), the carried tensors and the MoE output. The MoE of a
    // group grows the scheduler's buffer past what a ubatch graph takes by its rows times the MoE need per row.
    //   layer_exec_moe_tokens (0 = automatic): the token rows of one MoE bundle. The MoE of a layer runs once over the rows of a
    //     group when it fits, else in bundles of this many rows, each its own router and expert GEMMs, one after the other
    //     before the next layer. The expert-major op (MOE_FFN_HAD) takes the rows in passes of its own after its one
    //     router: there the value sets its pass rows (tokens * top-k) instead. Default: the largest multiple of the
    //     ubatch that keeps the MoE inside the scheduler buffer of a ubatch graph (at least one ubatch); the op keeps its
    //     own pass rows.
    //   layer_exec_group_tokens (0 = automatic): the token rows of a group (rounded down to ubatches). Default: the most ubatches
    //     whose edges and MoE fit the device budget, RANMA_LAYER_EXEC_DEV_MIB or half of the free device memory.
    //     A batch of more ubatches runs group after group: the experts of a layer are read once per group.
    const int     n_sets = one_set ? 1 : 2;
    const size_t  align  = std::max<size_t>(256, ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev)));
    size_t edge_pt = y_pt;
    for (size_t pt : slot_pt) {
        edge_pt += (size_t) n_sets*pt;
    }
    for (const auto & cs : carry_slots) {
        edge_pt += cs.first;
    }
    const size_t n_slots    = (size_t) n_sets*slot_pt.size() + carry_slots.size() + 1;
    const double moe_pt     = (double) moe_need_c/(double) C;
    const size_t sched_cur  = ggml_backend_sched_get_buffer_size(sc, backend_dev);
    const int64_t n_used    = (int64_t) model.hparams.n_expert_used();
    const int64_t moe_env   = lx_cfg_i64(&llama_layer_mode_config::layer_exec_moe_tokens);
    int64_t B       = 0; // graph bundle rows, 0: the rows of the group at once
    int64_t op_rows = 0; // the pass rows of the expert-major op, 0: its own
    if (fused) {
        op_rows = moe_env > 0 ? moe_env*n_used : 0;
    } else if (moe_env > 0) {
        B = moe_env;
    } else {
        B = C; // grown below to what the budget leaves
    }
    auto moe_extra = [&](int64_t w) -> size_t {
        const int64_t rows = B > 0 ? std::min(B, w) : w;
        const double need = moe_pt*(double) rows;
        size_t extra = need > (double) sched_cur ? (size_t) (need - (double) sched_cur) : 0;
        if (fused && op_rows > 0) {
            // the op's scratch for passes wider than the block the expert cache keeps for it comes from the device pool
            extra += ggml_moe_ffn_had_scratch_bytes(w, n_used, op_rows, (int64_t) model.hparams.n_expert, (int64_t) model.hparams.n_embd,
                (int64_t) model.hparams.n_ff_exp(), 0, 0);
        }
        return extra;
    };
    size_t dev_budget = 0;
    {
        const int64_t env = lx_env_i64("RANMA_LAYER_EXEC_DEV_MIB");
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        dev_budget = env > 0 ? (size_t) env << 20 : free/2;
    }
    size_t host_budget = 0;
    {
        const int64_t env = lx_env_i64("RANMA_LAYER_EXEC_HOST_MIB");
        if (env > 0) {
            host_budget = (size_t) env << 20;
        } else if (backend_cpu != nullptr) {
            size_t free = 0;
            size_t total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(backend_cpu), &free, &total);
            host_budget = free/2;
        }
    }
    auto group_fits = [&](int64_t g) {
        const int64_t w = g*C;
        const size_t dev_need  = edge_pt*(size_t) w + align*(n_slots + 1) + moe_extra(w);
        const size_t host_need = (input_bytes_max + meta_bytes_max)*(size_t) g;
        return (dev_budget == 0 || dev_need <= dev_budget) && (host_budget == 0 || host_need <= host_budget);
    };
    int64_t G = K;
    {
        const int64_t env = lx_cfg_i64(&llama_layer_mode_config::layer_exec_group_tokens);
        if (env > 0) {
            G = std::min<int64_t>(K, std::max<int64_t>(1, env/C));
        } else {
            while (G > 1 && !group_fits(G)) {
                G--;
            }
        }
    }
    if (G < 2 || !group_fits(G)) {
        n_outputs = save_n_outputs;
        if (log > 0) {
            LLAMA_LOG_INFO("%s: a group of %" PRId64 " ubatches of %" PRId64 " needs %zu MiB of device buffers (budget %zu MiB) "
                "and %zu MiB of host inputs and graphs (budget %zu MiB)\n", __func__, std::max<int64_t>(G, 2), C,
                (edge_pt*(size_t) (std::max<int64_t>(G, 2)*C) + moe_extra(std::max<int64_t>(G, 2)*C)) >> 20, dev_budget >> 20,
                ((input_bytes_max + meta_bytes_max)*(size_t) std::max<int64_t>(G, 2)) >> 20, host_budget >> 20);
        }
        return skip("not two ubatches fit the device and host budgets");
    }
    const int64_t n_groups = (K + G - 1)/G;
    const int64_t W_max    = std::min<int64_t>(N, G*C);
    if (!fused && moe_env <= 0) {
        // the widest bundle (in ubatches) the budget leaves beside the edges of the group, the MoE at once when it fits
        while (B < W_max && group_fits(G)) {
            B += C;
        }
        if (!group_fits(G)) {
            B -= C;
        }
        if (B >= W_max) {
            B = 0;
        }
    }

    // the MoE donors, one per width: the groups, and the bundles of a group
    std::map<int64_t, lx_donor> donors;
    auto bundle_widths = [&](int64_t w, std::vector<int64_t> & out) {
        out.clear();
        const int64_t b = B > 0 ? std::min(B, w) : w;
        for (int64_t r0 = 0; r0 < w; r0 += b) {
            out.push_back(std::min(b, w - r0));
        }
    };
    {
        std::set<int64_t> widths;
        std::vector<int64_t> bw;
        for (int64_t gi = 0; gi < n_groups; ++gi) {
            const int64_t k0 = gi*G;
            const int64_t k1 = std::min(K, k0 + G);
            const int64_t w  = std::min(N, k1*C) - k0*C;
            bundle_widths(w, bw);
            widths.insert(bw.begin(), bw.end());
        }
        for (int64_t w : widths) {
            trace("donor build", w);
            lx_donor & d = donors[w];
            d.ba.reset(new llama_batch_allocr(model.hparams.n_pos_per_embd()));
            d.res = build_reserve(w, *d.ba, mctx_full.get());
            trace("donor built", d.res ? ggml_graph_n_nodes(d.res->get_gf()) : -1);
            std::string err;
            if (!d.res || !llama_lx_regions(d.res->get_gf(), L, names, d.regions, d.inputs, err)) {
                n_outputs = save_n_outputs;
                ggml_backend_sched_reset(sc);
                if (log > 0) {
                    LLAMA_LOG_INFO("%s: the MoE of %" PRId64 " rows does not cut: %s\n", __func__, w, err.c_str());
                }
                return skip("the MoE regions do not cut");
            }
            for (ggml_tensor * t : d.inputs) {
                if (t != d.res->get_inp_tokens()) {
                    n_outputs = save_n_outputs;
                    ggml_backend_sched_reset(sc);
                    return skip("a MoE region reads an input other than the tokens");
                }
            }
            for (const auto & r : d.regions) {
                if (ggml_nbytes(r.y) != y_pt*(size_t) w) {
                    n_outputs = save_n_outputs;
                    ggml_backend_sched_reset(sc);
                    return skip("MoE outputs of different sizes");
                }
                lx_record_backends(sc, d.res->get_gf(), &r.nodes, d.backends, backend_dev);
            }
        }
    }
    ggml_backend_sched_reset(sc);
    mctx_full.reset();

    // ---- the buffers -----------------------------------------------------------------------------------------------
    ggml_backend_buffer_type_t buft_dev = ggml_backend_dev_buffer_type(dev);
    std::vector<size_t> slot_off;
    size_t edge_bytes = 0;
    for (int set = 0; set < n_sets; ++set) {
        for (size_t pt : slot_pt) {
            slot_off.push_back(edge_bytes);
            edge_bytes += lx_pad(pt*(size_t) W_max, align);
        }
    }
    std::vector<size_t> carry_off;
    for (const auto & cs : carry_slots) {
        carry_off.push_back(edge_bytes);
        edge_bytes += lx_pad(cs.first*(size_t) W_max, align);
    }
    const size_t y_off = edge_bytes;
    edge_bytes += lx_pad(y_pt*(size_t) W_max, align);
    edge_bytes += align;

    ggml_backend_buffer_ptr edge_buf(ggml_backend_buft_alloc_buffer(buft_dev, edge_bytes));
    if (!edge_buf) {
        n_outputs = save_n_outputs;
        return skip("cannot allocate the segment buffers");
    }
    char * edge_base = (char *) ggml_backend_buffer_get_base(edge_buf.get());
    char * y_base    = edge_base + y_off;

    ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
    if (buft_host == nullptr) {
        buft_host = ggml_backend_get_default_buffer_type(backend_cpu != nullptr ? backend_cpu : backend_dev);
    }

    // the token inputs of the donors
    for (auto & kv : donors) {
        lx_donor & d = kv.second;
        std::set<const ggml_tensor *> only(d.inputs.begin(), d.inputs.end());
        if (!only.empty()) {
            d.inp_buf.reset(lx_alloc_inputs(d.res->get_ctx(), buft_host, &only));
            if (!d.inp_buf) {
                n_outputs = save_n_outputs;
                return skip("cannot allocate the token input of a MoE");
            }
        }
    }

    const int64_t t_plan = ggml_time_us();
    if (log > 0) {
        LLAMA_LOG_INFO("layer_exec_plan: %d of %d layers layer by layer, %" PRId64 " ubatches of %" PRId64 " in %" PRId64 " group%s of "
            "up to %" PRId64 ", MoE %s, %.1f KiB of edges per row (%zu slot%s x %d set%s, %zu carried), MoE %.1f KiB per row, "
            "device budget %zu MiB, %zu MiB segment buffers\n",
            L, L_all, K, C, n_groups, n_groups > 1 ? "s" : "", G, fused ? (op_rows > 0 ? "expert-major op, passes of the rows set" :
            "expert-major op, its own passes") : B > 0 && B < W_max ? "in bundles" : "at once over a group", edge_pt/1024.0, slot_pt.size(),
            slot_pt.size() == 1 ? "" : "s", n_sets, n_sets > 1 ? "s" : "", carry_slots.size(), moe_pt/1024.0, dev_budget >> 20,
            edge_bytes >> 20);
        if (!fused && B > 0 && B < W_max) {
            LLAMA_LOG_INFO("layer_exec_plan: MoE bundles of %" PRId64 " rows\n", B);
        }
        if (fused && op_rows > 0) {
            LLAMA_LOG_INFO("layer_exec_plan: expert-major op passes of %" PRId64 " rows\n", op_rows);
        }
    }

    // ---- every group: the memory, graph and inputs of its ubatches, then its layers -----------------------------------
    // (from here on a failure removes the batch from the memory)
    const llama_pos pos0 = ub0.pos[0];
    auto abort_batch = [&](const char * what) {
        LLAMA_LOG_ERROR("%s: layer-major prefill failed: %s; removing the batch from the memory\n", __func__, what);
        ggml_backend_sched_synchronize(sc);
        ggml_backend_sched_reset(sc);
        n_outputs = save_n_outputs;
        memory->seq_rm(seq, pos0, -1);
        return -3;
    };

    const bool profiled = expert_iface && expert_backend && expert_profiled_bank != GGML_EXPERT_BANK_NONE &&
        (expert_profiled_seq == LLAMA_EXPERT_ALL_ROWS || expert_profiled_seq == seq);
    auto run = [&](lx_copy & cp, const std::vector<std::pair<const ggml_tensor *, ggml_backend_t>> & assigned,
                   int32_t n_tok) -> bool {
        ggml_backend_sched_reset(sc);
        for (const auto & a : assigned) {
            const auto it = cp.map.find(a.first);
            if (it != cp.map.end()) {
                ggml_backend_sched_set_tensor_backend(sc, it->second, a.second);
            }
        }
        ggml_backend_sched_set_alloc_compact(sc, true);
        const bool ok = ggml_backend_sched_alloc_graph(sc, cp.gf);
        ggml_backend_sched_set_alloc_compact(sc, false);
        if (!ok) {
            LLAMA_LOG_ERROR("%s: failed to allocate a segment graph\n", __func__);
            return false;
        }
        // as process_ubatch: the rows the expert profiler counts, and the prefetch of the table rows the graph gathers
        if (expert_iface && expert_backend) {
            expert_iface->profile_select(expert_backend, profiled ? 0 : -1, profiled ? n_tok : -1, expert_profiled_bank);
        }
        std::unique_ptr<llama_prefetch_batch> prefetch;
        if (n_tok > 1 && cparams.ple_prefetch != LLAMA_PLE_PREFETCH_OFF && !model.can_prefetch.empty()) {
            prefetch.reset(new llama_prefetch_batch(sc));
            prefetch->prepare(cp.gf, cparams.ple_prefetch, n_tok);
        }
        const ggml_status st = graph_compute(cp.gf, true, n_tok);
        if (prefetch) {
            prefetch->finish();
        }
        if (st != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_sched_synchronize(sc);
        return true;
    };

    // the rows of a slot are as wide as its widest tensor: the tensors of different segments in one slot keep the row
    // ranges of their ubatches apart
    auto slot_addr = [&](int s, size_t j, const llama_lx_edge & e, int64_t row0) {
        GGML_UNUSED(e);
        const int set = one_set ? 0 : (s & 1);
        return edge_base + slot_off[(size_t) set*slot_pt.size() + j] + (size_t) row0*slot_pt[j];
    };
    auto carry_addr = [&](size_t k, int64_t row0) {
        return edge_base + carry_off[k] + (size_t) row0*carry_slots[k].first;
    };

    const int64_t n_vocab = model.vocab.n_tokens();
    int64_t n_outputs_prev = 0;
    int64_t t_seg_us = 0;
    int64_t t_moe_us = 0;
    int64_t t_inputs_us = 0;
    int64_t t_build_us = 0;
    int64_t n_graphs = 0;
    int64_t n_moe = 0;
    size_t inputs_held_max = 0; // the input buffers of the ubatches of a group
    size_t inputs_copied = 0;   // the inputs the segments read, summed over segments (copied once per segment when on the host)
    // the attention masks per layer kind (SWA only; compressed; HCA, ratio 128 and up), counted per layer and ubatch: made
    // on the device, or read from host inputs; and the last segment of the layers after the tail start
    int mask_dev[3]  = { 0, 0, 0 };
    int mask_host[3] = { 0, 0, 0 };
    int rest_dev = 0;
    int rest_host = 0;
    size_t host_masks_held = 0;
    size_t host_masks_read = 0;

    int64_t k_glob = 0;   // ubatches of the batch taken from the memory context
    int64_t i0_glob = 0;  // their tokens
    for (int64_t gi = 0; gi < n_groups; ++gi) {
        const int64_t t_g0 = ggml_time_us();
        const int64_t k0 = gi*G;
        const int64_t kn = std::min(K, k0 + G) - k0;
        const int64_t g0 = i0_glob;   // the first token of the group in the batch

        std::vector<lx_chunk> chunks((size_t) kn);
        std::vector<llama_token> tokens;
        tokens.reserve((size_t) W_max);
        int64_t i0 = 0;
        size_t inputs_held = 0;
        for (int64_t k = 0; k < kn; ++k, ++k_glob) {
            if (k_glob > 0 && !mctx->next()) {
                return abort_batch("fewer ubatches than planned");
            }
            if (!mctx->apply()) {
                return abort_batch("failed to apply the memory context");
            }
            const llama_ubatch & ub = mctx->get_ubatch();
            auto & c = chunks[(size_t) k];
            c.i0 = i0;
            c.n  = ub.n_tokens;
            i0  += ub.n_tokens;
            if (i0 > W_max || g0 + i0 > N || ub.token == nullptr) {
                return abort_batch("unexpected ubatch");
            }
            tokens.insert(tokens.end(), ub.token, ub.token + ub.n_tokens);

            int32_t n_out = 0;
            if (n_outputs_all == n_tokens_all) {
                n_out = (int32_t) ub.n_tokens;
            } else {
                for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                    n_out += ub.output[i] != 0;
                }
            }
            c.n_outputs = n_out;
            n_outputs = n_out;

            c.res.reset(new llm_graph_result(graph_max_nodes(ub.n_tokens)));
            const auto gparams = graph_params(c.res.get(), ub, mctx, LLM_GRAPH_TYPE_DEFAULT);
            c.res->reset();
            ggml_backend_sched_reset(sc);
            ggml_cgraph * gf = model.build_graph(gparams);
            if (gf == nullptr) {
                return abort_batch("failed to build a chunk graph");
            }
            c.res->trim();
            lx_record_backends(sc, gf, nullptr, c.backends, backend_dev);
            ggml_backend_sched_reset(sc);

            trace("chunk built", k_glob, ggml_graph_n_nodes(gf));
            if (!llama_lx_analyze(gf, L, names, ub.n_tokens, c.plan)) {
                LLAMA_LOG_ERROR("%s: chunk %" PRId64 ": %s\n", __func__, k_glob, c.plan.err.c_str());
                return abort_batch("a chunk graph does not cut like the reserve graph");
            }
            for (int s = 0; s <= L; ++s) {
                if (c.plan.out[s].size() > slot_pt.size()) {
                    return abort_batch("a chunk graph has more segment outputs than planned");
                }
                for (size_t j = 0; j < c.plan.out[s].size(); ++j) {
                    if (c.plan.out[s][j].pt > slot_pt[j]) {
                        return abort_batch("a segment output is wider than planned");
                    }
                }
            }
            {
                std::map<int, int> ord;
                for (const auto & e : c.plan.carry) {
                    const auto it = carry_plan.find({ e.s, ord[e.s]++ });
                    if (it == carry_plan.end() || e.t_last > it->second.t_last || e.pt > carry_slots[(size_t) it->second.slot].first) {
                        return abort_batch("a carried tensor of a chunk graph does not fit the plan");
                    }
                }
            }
            // only the inputs the segments read: the others (e.g. the host masks a chunked graph makes on the device) stay
            // unallocated, as in a graph the scheduler allocates
            {
                std::set<const ggml_tensor *> used(c.plan.inputs.begin(), c.plan.inputs.end());
                c.inp_buf.reset(lx_alloc_inputs(c.res->get_ctx(), buft_host, &used, gf));
            }
            if (!c.inp_buf) {
                return abort_batch("cannot allocate the inputs of a chunk");
            }
            const int64_t t_in = ggml_time_us();
            c.res->set_inputs(&ub);
            t_inputs_us += ggml_time_us() - t_in;
            inputs_held += ggml_backend_buffer_get_size(c.inp_buf.get());
            for (size_t b : c.plan.seg_input_bytes) {
                inputs_copied += b;
            }
            host_masks_held += c.plan.host_mask_bytes;
            for (int s = 0; s <= L; ++s) {
                host_masks_read += c.plan.seg_host_mask_bytes[(size_t) s];
                const bool host = c.plan.seg_host_mask_bytes[(size_t) s] > 0;
                const bool devm = c.plan.seg_dev_masks[(size_t) s] > 0;
                if (s == L) {
                    rest_host += host;
                    rest_dev  += !host && devm;
                    continue;
                }
                const uint32_t ratio = model.hparams.dsv4_compress_ratios[s];
                const int kind = ratio == 0 ? 0 : ratio >= 128 ? 2 : 1;
                if (host) {
                    mask_host[kind]++;
                } else if (devm) {
                    mask_dev[kind]++;
                }
            }
            if (!c.plan.one_set && one_set) {
                return abort_batch("a chunk graph needs two buffer sets");
            }
        }
        const int64_t W = i0;
        inputs_held_max = std::max(inputs_held_max, inputs_held);
        i0_glob += W;
        t_build_us += ggml_time_us() - t_g0;

        std::vector<int64_t> bw;
        bundle_widths(W, bw);
        for (int64_t w : bw) {
            if (donors.count(w) == 0) {
                return abort_batch("no MoE graph of a bundle width");
            }
        }


        // ---- the layers of the group ---------------------------------------------------------------------------------
        for (int l = 0; l <= L; ++l) {
            // segment l of every chunk
            const int64_t t0 = ggml_time_us();
            for (auto & c : chunks) {
                const llama_lx_plan & p = c.plan;
                const lx_graph g(c.res->get_gf());
                if (l == L && c.n_outputs == 0) {
                    // the last segment ends the last layer run layer by layer, runs the layers after it and makes the
                    // logits: nothing of a chunk without outputs survives it unless it writes a state
                    bool writes = false;
                    for (int i : p.nodes[l]) {
                        const ggml_tensor * t = g.node(i);
                        writes = writes || (!lx_is_view(t->op) && t->view_src != nullptr && g.kind(t->view_src) == LX_STATE);
                    }
                    if (!writes) {
                        continue;
                    }
                }
                std::vector<const llama_lx_edge *> carried_in;
                std::vector<const llama_lx_edge *> carried_out;
                std::vector<size_t> carried_in_slot;
                std::vector<size_t> carried_out_slot;
                {
                    std::map<int, int> ord;
                    for (const auto & e : p.carry) {
                        const auto & cpl = carry_plan.at({ e.s, ord[e.s]++ });
                        if (e.s < l && l <= e.t_last) {
                            carried_in.push_back(&e);
                            carried_in_slot.push_back((size_t) cpl.slot);
                        } else if (e.s == l) {
                            carried_out.push_back(&e);
                            carried_out_slot.push_back((size_t) cpl.slot);
                        }
                    }
                }
                lx_copy cp;
                const size_t n_in = (l > 0 ? p.out[l - 1].size() + 1 : 0) + carried_in.size();
                if (!cp.init(p.nodes[l].size() + n_in, p.nodes[l].size()*4 + n_in + 256)) {
                    return abort_batch("cannot make a segment graph");
                }
                // the inputs: the outputs of the previous segment, the MoE output and the carried tensors it reads
                if (l > 0) {
                    for (size_t j = 0; j < p.out[l - 1].size(); ++j) {
                        const auto & e = p.out[l - 1][j];
                        if (cp.leaf(e.t, edge_buf.get(), slot_addr(l - 1, j, e, c.i0), e.place_as) == nullptr) {
                            return abort_batch("cannot place a segment input");
                        }
                    }
                    const ggml_tensor * y = p.y[l - 1];
                    if (ggml_nbytes(y) != y_pt*(size_t) c.n) {
                        return abort_batch("the MoE output rows of a chunk differ from the MoE graph");
                    }
                    if (cp.leaf(y, edge_buf.get(), y_base + (size_t) c.i0*y_pt) == nullptr) {
                        return abort_batch("cannot place a MoE output");
                    }
                }
                for (size_t k = 0; k < carried_in.size(); ++k) {
                    const auto & e = *carried_in[k];
                    if (cp.leaf(e.t, edge_buf.get(), carry_addr(carried_in_slot[k], c.i0), e.place_as) == nullptr) {
                        return abort_batch("cannot place a carried tensor");
                    }
                }
                // the outputs, computed in place
                std::map<const ggml_tensor *, std::pair<ggml_backend_buffer_t, char *>> place;
                if (l < L) {
                    for (size_t j = 0; j < p.out[l].size(); ++j) {
                        const auto & e = p.out[l][j];
                        place[e.place_as != nullptr ? e.place_as : e.t] = { edge_buf.get(), slot_addr(l, j, e, c.i0) };
                    }
                }
                for (size_t k = 0; k < carried_out.size(); ++k) {
                    const auto & e = *carried_out[k];
                    place[e.place_as != nullptr ? e.place_as : e.t] =
                        { edge_buf.get(), carry_addr(carried_out_slot[k], c.i0) };
                }
                std::string err;
                if (!cp.copy(g, p.nodes[l], place, err)) {
                    LLAMA_LOG_ERROR("%s: %s\n", __func__, err.c_str());
                    return abort_batch("cannot copy a segment");
                }
                trace("segment", l, g0 + c.i0);
                if (!run(cp, c.backends, (int32_t) c.n)) {
                    return abort_batch("a segment failed");
                }
                n_graphs++;

                if (l == L && c.n_outputs > 0) {
                    ggml_tensor * t_logits = c.res->get_logits();
                    const auto it = t_logits != nullptr ? cp.map.find(t_logits) : cp.map.end();
                    if (logits.data != nullptr && it != cp.map.end()) {
                        GGML_ASSERT((n_outputs_prev + c.n_outputs)*n_vocab <= (int64_t) logits.size);
                        ggml_backend_tensor_get(it->second, logits.data + n_outputs_prev*n_vocab, 0,
                                (size_t) c.n_outputs*n_vocab*sizeof(float));
                    }
                    n_outputs_prev += c.n_outputs;
                }
            }
            const int64_t t1 = ggml_time_us();
            t_seg_us += t1 - t0;
            if (l == L) {
                break;
            }

            // the MoE of layer l over every row of the group, in bundles where they do not fit at once
            // the router input: the slot of x[l] in the outputs of segment l (the same slot in every chunk)
            const auto & out0 = chunks[0].plan.out[(size_t) l];
            size_t jx = SIZE_MAX;
            for (size_t j = 0; j < out0.size(); ++j) {
                if (out0[j].t == chunks[0].plan.x[(size_t) l]) {
                    jx = j;
                }
            }
            for (const auto & c : chunks) {
                const auto & o = c.plan.out[(size_t) l];
                if (jx >= o.size() || o[jx].t != c.plan.x[(size_t) l] || o[jx].pt != out0[jx].pt) {
                    return abort_batch("the router input is not in the same slot of every chunk");
                }
            }
            int64_t r0 = 0;
            for (int64_t w : bw) {
                lx_donor & d = donors.at(w);
                const llama_lx_region & r = d.regions[(size_t) l];
                if (out0[jx].pt != slot_pt[jx] || out0[jx].pt*(size_t) w != ggml_nbytes(r.x)) {
                    return abort_batch("the router input rows differ from the MoE graph");
                }
                const lx_graph gd(d.res->get_gf());
                lx_copy cp;
                if (!cp.init(r.nodes.size() + 1, r.nodes.size()*4 + 256)) {
                    return abort_batch("cannot make a MoE graph");
                }
                if (cp.leaf(r.x, edge_buf.get(), slot_addr(l, jx, out0[jx], r0)) == nullptr) {
                    return abort_batch("cannot place the router input");
                }
                std::map<const ggml_tensor *, std::pair<ggml_backend_buffer_t, char *>> place;
                place[r.y] = { edge_buf.get(), y_base + (size_t) r0*y_pt };
                std::string err;
                if (!cp.copy(gd, r.nodes, place, err)) {
                    LLAMA_LOG_ERROR("%s: %s\n", __func__, err.c_str());
                    return abort_batch("cannot copy a MoE region");
                }
                if (op_rows > 0) {
                    for (int i : r.nodes) {
                        ggml_tensor * t = cp.map.at(gd.node(i));
                        if (t->op == GGML_OP_MOE_FFN_HAD) {
                            ((int32_t *) t->op_params)[2] = (int32_t) op_rows;
                        }
                    }
                }
                if (d.inp_buf) {
                    ggml_tensor * t = d.res->get_inp_tokens();
                    if (t == nullptr || ggml_nelements(t) != w) {
                        return abort_batch("the token input of a MoE");
                    }
                    ggml_backend_tensor_set(t, tokens.data() + r0, 0, (size_t) w*sizeof(llama_token));
                }
                trace("moe", l, w);
                if (!run(cp, d.backends, (int32_t) w)) {
                    return abort_batch("a MoE failed");
                }
                n_graphs++;
                n_moe++;
                r0 += w;
            }
            t_moe_us += ggml_time_us() - t1;
        }
    }

    if (k_glob != K || i0_glob != N) {
        return abort_batch("the ubatches do not cover the batch");
    }
    if ((uint32_t) n_outputs_prev != n_outputs_all) {
        return abort_batch("the outputs do not add up");
    }

    // the call is complete: the memory context ends it as after its last ubatch
    if (mctx->next()) {
        return abort_batch("more ubatches than planned");
    }

    ggml_backend_sched_reset(sc);
    n_outputs = save_n_outputs;

    if (log > 0) {
        const int64_t t_end = ggml_time_us();
        LLAMA_LOG_INFO("%s: %" PRId64 " tokens in %" PRId64 " chunks of %" PRId64 ", %d layers, %" PRId64 " graphs (%" PRId64 " MoE): plan %.1f ms, "
            "chunk graphs %.1f ms (inputs %.1f ms), segments %.1f ms, MoE %.1f ms, total %.1f ms; %zu MiB segment buffers (%d set%s, "
            "%zu carried), %" PRId64 " group%s, %.1f MiB inputs held, %.1f MiB inputs read by the segments\n", __func__, N, K, C, L,
            n_graphs, n_moe, (t_plan - t_start)/1000.0, t_build_us/1000.0, t_inputs_us/1000.0, t_seg_us/1000.0, t_moe_us/1000.0,
            (t_end - t_start)/1000.0, edge_bytes >> 20, n_sets, n_sets > 1 ? "s" : "", carry_slots.size(), n_groups,
            n_groups > 1 ? "s" : "", inputs_held_max/1048576.0, inputs_copied/1048576.0);
        // one line a log check can match: "layer_exec_masks: device|host|mixed ..."
        const int n_dev  = mask_dev[0] + mask_dev[1] + mask_dev[2] + rest_dev;
        const int n_host = mask_host[0] + mask_host[1] + mask_host[2] + rest_host;
        std::string rest;
        if (L < L_all) {
            rest = format(", last segment (layers %d..%d) %d/%d", L, L_all - 1, rest_dev, rest_host);
        }
        LLAMA_LOG_INFO("layer_exec_masks: %s, layers x chunks device/host: SWA %d/%d, CSA %d/%d, HCA %d/%d%s; host masks %.1f MiB held, "
            "%.1f MiB read by the segments\n", n_host == 0 ? "device" : n_dev == 0 ? "host" : "mixed",
            mask_dev[0], mask_host[0], mask_dev[1], mask_host[1], mask_dev[2], mask_host[2], rest.c_str(),
            host_masks_held/1048576.0, host_masks_read/1048576.0);
    }

    return 0;
}
