#include "ngram-varlen.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <functional>

//
// common_ngram_varlen
//

size_t common_ngram_varlen::key3_hash::operator()(const key3 & k) const {
    uint64_t h = (uint32_t) k.t0;
    h = h*0x9E3779B97F4A7C15ULL + (uint32_t) k.t1;
    h = h*0x9E3779B97F4A7C15ULL + (uint32_t) k.t2;
    return (size_t) (h ^ (h >> 29));
}

common_ngram_varlen::common_ngram_varlen(const common_ngram_varlen_params & params, uint32_t n_seq) :
    params(params), seqs(n_seq) {
}

uint64_t common_ngram_varlen::hash_run(const llama_token * tokens) const {
    uint64_t h = 1469598103934665603ULL;
    for (int32_t i = 0; i < params.n_dedup; ++i) {
        h = (h ^ (uint32_t) tokens[i])*1099511628211ULL;
    }
    return h;
}

bool common_ngram_varlen::active(llama_seq_id seq_id) const {
    return seq_id >= 0 && (size_t) seq_id < seqs.size() && seqs[seq_id].active;
}

size_t common_ngram_varlen::n_prompt(llama_seq_id seq_id) const {
    return seqs[seq_id].n_prompt;
}

size_t common_ngram_varlen::n_generated(llama_seq_id seq_id) const {
    return seqs[seq_id].history.size() - seqs[seq_id].n_prompt;
}

const std::vector<llama_token> & common_ngram_varlen::history(llama_seq_id seq_id) const {
    return seqs[seq_id].history;
}

size_t common_ngram_varlen::get_n_samples() const {
    size_t n = 0;
    for (const auto & [_, items] : index) {
        n += items.size();
    }
    return n;
}

// the prompt positions that hold new text: not in the longest prefix shared with an earlier request, and not in a
// run of at least n_dedup tokens equal to an earlier request
std::vector<uint8_t> common_ngram_varlen::new_mask(const std::vector<llama_token> & seq) const {
    const size_t n = seq.size();
    const size_t m = (size_t) params.n_dedup;

    std::vector<uint8_t> fresh(n, 1);

    size_t prefix = 0;
    for (size_t r = prior_first; r < prior.size(); ++r) {
        const auto & old = prior[r];
        const size_t n_max = std::min(old.size(), n);
        size_t k = 0;
        while (k < n_max && old[k] == seq[k]) {
            ++k;
        }
        prefix = std::max(prefix, k);
    }
    std::fill(fresh.begin(), fresh.begin() + prefix, 0);

    size_t p = prefix;
    while (m > 0 && p + m <= n) {
        size_t best = 0;
        const auto it = seen16.find(hash_run(seq.data() + p));
        if (it != seen16.end()) {
            for (const uint64_t e : it->second) {
                const size_t r = (size_t) (e >> 32);
                const size_t j = (size_t) (e & 0xFFFFFFFFu);
                if (r < prior_first) {
                    continue;
                }
                const auto & old = prior[r];
                if (!std::equal(seq.begin() + p, seq.begin() + p + m, old.begin() + j)) {
                    continue; // hash collision
                }
                size_t k = m;
                while (p + k < n && j + k < old.size() && seq[p + k] == old[j + k]) {
                    ++k;
                }
                best = std::max(best, k);
            }
        }
        if (best > 0) {
            std::fill(fresh.begin() + p, fresh.begin() + p + best, 0);
            p += best;
        } else {
            p += 1;
        }
    }

    return fresh;
}

void common_ngram_varlen::begin(llama_seq_id seq_id, const std::vector<llama_token> & prompt) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seqs.size());

    if (seqs[seq_id].active) {
        end(seq_id);
    }

    const std::vector<uint8_t> fresh = new_mask(prompt);

    // each run of new text is a segment of prompt samples
    size_t j = 0;
    while (j < prompt.size()) {
        if (!fresh[j]) {
            ++j;
            continue;
        }
        size_t e = j + 1;
        while (e < prompt.size() && fresh[e]) {
            ++e;
        }

        const int32_t sid = (int32_t) segments.size();
        segments.push_back({ std::vector<llama_token>(prompt.begin() + j, prompt.begin() + e), true, clock + 1, false });

        const auto & toks = segments.back().tokens;
        for (size_t p = 2; p < toks.size(); ++p) {
            index[{ toks[p - 2], toks[p - 1], toks[p] }].push_back({ sid, (int32_t) p });
        }
        clock += (int64_t) toks.size();
        j = e;
    }

    auto & st = seqs[seq_id];
    st.active   = true;
    st.n_prompt = prompt.size();
    st.history  = prompt;
    st.sid      = (int32_t) segments.size();

    segments.push_back({ {}, false, clock + 1, true });
}

void common_ngram_varlen::add(llama_seq_id seq_id, llama_token token) {
    auto & st = seqs[seq_id];
    GGML_ASSERT(st.active);

    st.history.push_back(token);

    auto & toks = segments[st.sid].tokens;
    toks.push_back(token);
    clock += 1;

    const size_t n = toks.size();
    if (n >= 3) {
        index[{ toks[n - 3], toks[n - 2], toks[n - 1] }].push_back({ st.sid, (int32_t) (n - 1) });
    }
}

void common_ngram_varlen::rebuild_seen() {
    seen16.clear();
    const size_t m = (size_t) params.n_dedup;
    for (size_t r = prior_first; r < prior.size(); ++r) {
        const auto & old = prior[r];
        for (size_t j = 0; m > 0 && j + m <= old.size(); ++j) {
            seen16[hash_run(old.data() + j)].push_back(((uint64_t) r << 32) | (uint64_t) j);
        }
    }
    prior_dropped = 0;
}

void common_ngram_varlen::add_prior(const std::vector<llama_token> & tokens) {
    const size_t r = prior.size();
    prior.push_back(tokens);
    prior_tokens += (int64_t) tokens.size();

    const size_t m = (size_t) params.n_dedup;
    for (size_t j = 0; m > 0 && j + m <= tokens.size(); ++j) {
        seen16[hash_run(tokens.data() + j)].push_back(((uint64_t) r << 32) | (uint64_t) j);
    }

    // bound the memory of a long session: the oldest requests leave the new-text test
    bool dropped = false;
    while (prior_tokens > params.n_prior_max && prior_first + 1 < prior.size()) {
        prior_tokens  -= (int64_t) prior[prior_first].size();
        prior_dropped += (int64_t) prior[prior_first].size();
        prior[prior_first] = {};
        prior_first++;
        dropped = true;
    }
    if (dropped && prior_dropped > prior_tokens) {
        rebuild_seen();
    }
}

void common_ngram_varlen::purge() {
    const int64_t window = params.window;

    for (auto it = index.begin(); it != index.end();) {
        auto & items = it->second;
        items.erase(std::remove_if(items.begin(), items.end(), [&](const entry & e) {
            return clock - (segments[e.sid].base + e.p) >= window;
        }), items.end());
        if (items.empty()) {
            it = index.erase(it);
        } else {
            ++it;
        }
    }

    // a segment whose newest sample expired is no longer read
    for (auto & seg : segments) {
        if (!seg.live && !seg.tokens.empty() && clock - (seg.base + (int64_t) seg.tokens.size() - 1) >= window) {
            seg.tokens = {};
        }
    }
}

void common_ngram_varlen::end(llama_seq_id seq_id) {
    auto & st = seqs[seq_id];
    if (!st.active) {
        return;
    }

    segments[st.sid].live = false;
    add_prior(st.history);

    st.active = false;
    st.history.clear();
    st.n_prompt = 0;
    st.sid = -1;

    purge();
}

common_ngram_varlen_draft common_ngram_varlen::draft(llama_seq_id seq_id, const common_ngram_varlen_draft_opts * opts) const {
    common_ngram_varlen_draft res;

    if (!active(seq_id)) {
        return res;
    }

    const auto & ctx = seqs[seq_id].history;
    const size_t n_ctx = ctx.size();
    if (n_ctx < 3) {
        return res;
    }

    const auto it = index.find({ ctx[n_ctx - 3], ctx[n_ctx - 2], ctx[n_ctx - 1] });
    if (it == index.end()) {
        return res;
    }

    struct match {
        int32_t n;
        int32_t sid;
        int32_t start;
        double  w;
    };

    const int64_t window = params.window;

    std::vector<match> matches;
    matches.reserve(it->second.size());
    for (const entry & e : it->second) {
        const segment & seg = segments[e.sid];
        const int64_t age = clock - (seg.base + e.p);
        if (age + 3 > window || (size_t) e.p + 1 >= seg.tokens.size()) {
            continue;
        }
        // the matched context, backwards from the 3-gram; seg.tokens only holds tokens already seen
        const int64_t n_max = std::min<int64_t>({ (int64_t) e.p + 1, (int64_t) n_ctx, window - age });
        int64_t n = 3;
        while (n < n_max && seg.tokens[e.p - n] == ctx[n_ctx - n - 1]) {
            ++n;
        }
        const double w = std::pow(2.0, -(double) age/params.half_life)*(seg.prompt ? params.w_prompt : 1.0);
        matches.push_back({ (int32_t) n, e.sid, e.p + 1, w });
    }

    std::vector<int32_t> ns;
    ns.reserve(matches.size());
    for (const auto & m : matches) {
        ns.push_back(m.n);
    }
    std::sort(ns.begin(), ns.end(), std::greater<int32_t>());
    ns.erase(std::unique(ns.begin(), ns.end()), ns.end());

    const size_t n_support = (size_t) std::max(2, params.n_support);
    const bool   meta      = opts && (opts->meta || opts->alt || opts->shadow);

    std::vector<int32_t> active;
    std::vector<int32_t> keep;
    std::vector<std::pair<llama_token, double>> counts;

    // the draft that follows the cohort of the samples that match at least n tokens
    auto walk = [&](int32_t n, size_t min_support, bool ladder, double p_min, size_t cap, int32_t & n_cohort,
            std::vector<double> * raw, std::vector<int32_t> * sup) {
        std::vector<llama_token> path;

        // the cohort: every sample that matches at least n tokens, in sample order
        active.clear();
        for (int32_t k = 0; k < (int32_t) matches.size(); ++k) {
            if (matches[k].n >= n) {
                active.push_back(k);
            }
        }
        n_cohort = (int32_t) active.size();
        if ((size_t) n_cohort < min_support) {
            return path;
        }

        // the weight of the cohort, summed as the replay does (Python 3.12 sum(): Neumaier compensation)
        double total = matches[active[0]].w;
        double comp  = 0.0;
        for (size_t k = 1; k < active.size(); ++k) {
            const double x = matches[active[k]].w;
            const double t = total + x;
            comp += std::fabs(total) >= std::fabs(x) ? (total - t) + x : (x - t) + total;
            total = t;
        }
        if (comp != 0.0 && std::isfinite(comp)) {
            total += comp;
        }

        for (size_t j = 0; !active.empty(); ++j) {
            counts.clear();
            for (const int32_t k : active) {
                const auto & toks = segments[matches[k].sid].tokens;
                const size_t i = (size_t) matches[k].start + j;
                if (i >= toks.size()) {
                    continue;
                }
                auto c = std::find_if(counts.begin(), counts.end(), [&](const auto & x) { return x.first == toks[i]; });
                if (c == counts.end()) {
                    counts.push_back({ toks[i], matches[k].w });
                } else {
                    c->second += matches[k].w;
                }
            }
            if (counts.empty()) {
                break;
            }

            // the heaviest continuation, the lowest token id on a tie
            auto best = counts.begin();
            for (auto c = counts.begin() + 1; c != counts.end(); ++c) {
                if (c->second > best->second || (c->second == best->second && c->first < best->first)) {
                    best = c;
                }
            }
            const llama_token token = best->first;
            const double prob = best->second/total;
            if (prob + 1e-12 < p_min) {
                break;
            }

            keep.clear();
            for (const int32_t k : active) {
                const auto & toks = segments[matches[k].sid].tokens;
                const size_t i = (size_t) matches[k].start + j;
                if (i < toks.size() && toks[i] == token) {
                    keep.push_back(k);
                }
            }
            active.swap(keep);

            const size_t support = active.size();
            if (support < min_support || (ladder && (int64_t) j + 1 > (int64_t) params.ladder*(int64_t) (support - 1))) {
                break;
            }
            path.push_back(token);
            if (raw) {
                raw->push_back(std::min(1.0, prob));
                sup->push_back((int32_t) support);
            }
            if (path.size() >= cap) {
                break;
            }
        }

        return path;
    };

    if (opts && opts->shadow && !ns.empty()) {
        // the representative cohorts of the replay: longest n, the longest n whose first token has 2 samples, shortest n
        const size_t cap = (size_t) std::max(1, opts->shadow_cap);
        auto cohort = [&](int32_t n) {
            common_ngram_varlen_cand c;
            auto path = walk(n, 1, false, 0.0, cap, c.n_samples, &c.raw, &c.support);
            c.tokens = std::move(path);
            c.n      = n;
            return c;
        };
        auto have = [&](int32_t n) {
            return std::any_of(res.shadow.begin(), res.shadow.end(), [&](const auto & c) { return c.n == n; });
        };
        auto first = cohort(ns.front());
        bool two = !first.tokens.empty() && first.support[0] >= 2;
        if (!first.tokens.empty()) {
            res.shadow.push_back(std::move(first));
        }
        for (size_t k = 1; !two && k < ns.size(); ++k) {
            auto c = cohort(ns[k]);
            if (!c.tokens.empty() && c.support[0] >= 2) {
                two = true;
                res.shadow.push_back(std::move(c));
            }
        }
        if (!have(ns.back())) {
            auto c = cohort(ns.back());
            if (!c.tokens.empty()) {
                res.shadow.push_back(std::move(c));
            }
        }
    }

    for (const int32_t n : ns) {
        if (n < params.n_min) {
            break;
        }

        int32_t n_cohort = 0;
        std::vector<double>  raw;
        std::vector<int32_t> sup;
        auto path = walk(n, n_support, true, params.p_min, SIZE_MAX, n_cohort, meta ? &raw : nullptr, &sup);

        if (!path.empty()) {
            res.tokens    = std::move(path);
            res.n         = n;
            res.n_samples = n_cohort;
            res.raw       = std::move(raw);
            res.support   = std::move(sup);
            return res;
        }
    }

    if (opts && opts->alt) {
        const auto & ap = opts->alt_params;
        for (const int32_t n : ns) {
            if (n < std::max(params.n_min, ap.n_min)) {
                break;
            }

            common_ngram_varlen_cand cand;
            auto path = walk(n, 1, false, ap.raw_min, (size_t) std::max(1, ap.max_len), cand.n_samples, &cand.raw, &cand.support);
            if (!path.empty()) {
                cand.tokens = std::move(path);
                cand.n      = n;
                res.alt.push_back(std::move(cand));
            }
        }
    }

    return res;
}

//
// common_ngram_varlen_hazard
//

common_ngram_varlen_hazard::common_ngram_varlen_hazard(const common_ngram_varlen_hazard_params & params) : params(params) {
    GGML_ASSERT(!this->params.cost.empty() && this->params.width >= 1);
}

static int32_t hazard_pos_bucket(int32_t j) {
    return j <= 4 ? j : j <= 8 ? 5 : j <= 16 ? 6 : 7;
}

static int32_t hazard_support_bucket(int32_t s) {
    return s == 1 ? 0 : s == 2 ? 1 : s <= 4 ? 2 : s <= 8 ? 3 : 4;
}

static int32_t hazard_n_bucket(int32_t n) {
    return n < 5 ? 0 : n < 8 ? 1 : n < 16 ? 2 : 3;
}

static int32_t hazard_raw_bucket(double p) {
    return p < 0.5 ? 0 : p < 0.8 ? 1 : p < 0.95 ? 2 : 3;
}

uint32_t common_ngram_varlen_hazard::key_global(int32_t j, int32_t support) {
    return (uint32_t) (hazard_pos_bucket(j)*8 + hazard_support_bucket(support));
}

uint32_t common_ngram_varlen_hazard::key_cell(int32_t j, const feat & f) {
    return (uint32_t) (((hazard_pos_bucket(j)*4 + hazard_n_bucket(f.n))*8 + hazard_support_bucket(f.support))*4 + hazard_raw_bucket(f.raw));
}

double common_ngram_varlen_hazard::cost(size_t rows) const {
    return cost_at(params.cost, rows);
}

double common_ngram_varlen_hazard::cost_at(const std::vector<double> & c, size_t rows) {
    if (c.empty()) {
        return 0.0;
    }
    if (rows >= 1 && rows <= c.size()) {
        return c[rows - 1];
    }
    const double step = c.size() > 1 ? c.back() - c[c.size() - 2] : 0.0;
    return c.back() + step*(double) (rows - c.size());
}

void common_ngram_varlen_hazard::read(const std::unordered_map<uint32_t, cell> & table, uint32_t key, int64_t clock,
        double & s, double & c) const {
    const auto it = table.find(key);
    if (it == table.end()) {
        s = c = 0.0;
        return;
    }
    const double f = std::pow(2.0, -(double) std::max<int64_t>(0, clock - it->second.clock)/params.half_life);
    s = it->second.s*f;
    c = it->second.c*f;
}

void common_ngram_varlen_hazard::add(const feat & f, int32_t j, int y, double w, int64_t clock) {
    double s, c;

    read(cells, key_cell(j, f), clock, s, c);
    cells[key_cell(j, f)] = { s + w*y, c + w, clock };

    read(global, key_global(j, f.support), clock, s, c);
    global[key_global(j, f.support)] = { s + w*y, c + w, clock };
}

void common_ngram_varlen_hazard::shadow_add(shadow_list & list, const std::vector<common_ngram_varlen_cand> & picks,
        size_t n_hist, int64_t clock) {
    if (picks.empty()) {
        return;
    }
    shadow_anchor a;
    a.clock0 = clock;
    a.n_hist = n_hist;
    for (const auto & c : picks) {
        shadow_pick p;
        p.tokens = c.tokens;
        p.w      = 1.0/(double) picks.size();
        for (size_t j = 0; j < c.tokens.size(); ++j) {
            p.feats.push_back({ c.n, c.raw[j], c.support[j] });
        }
        a.picks.push_back(std::move(p));
    }
    list.push_back(std::move(a));
}

void common_ngram_varlen_hazard::shadow_step(shadow_list & list, const std::vector<llama_token> & hist, bool end) {
    for (auto & a : list) {
        for (auto & p : a.picks) {
            while (!p.fin) {
                const size_t j = p.y.size();
                if (a.n_hist + j >= hist.size()) {
                    break;
                }
                const bool hit = p.tokens[j] == hist[a.n_hist + j];
                p.y.push_back(hit ? 1 : 0);
                if (!hit || p.y.size() == p.tokens.size()) {
                    p.fin = true;
                }
            }
            if (end) {
                p.fin = true;
            }
        }
    }
}

size_t common_ngram_varlen_hazard::shadow_flush(shadow_list & list, int64_t clock) {
    size_t n = 0;
    for (auto it = list.begin(); it != list.end();) {
        size_t last = 0;
        bool   done = true;
        for (const auto & p : it->picks) {
            done = done && p.fin;
            last = std::max(last, p.y.size());
        }
        const int64_t ready = it->clock0 + (int64_t) last;
        if (!done || ready > clock) {
            ++it;
            continue;
        }
        for (const auto & p : it->picks) {
            for (size_t j = 0; j < p.y.size(); ++j) {
                add(p.feats[j], (int32_t) j + 1, p.y[j], p.w, ready);
                n++;
            }
        }
        it = list.erase(it);
    }
    return n;
}

// the cost per output token of a draft with these P_j, in pieces of width tokens
double common_ngram_varlen_hazard::expected(const std::vector<double> & ps, size_t length, const std::vector<double> & c,
        size_t width) const {
    double cost_sum = 0.0;
    double out_sum  = 0.0;
    for (size_t offset = 0; offset < length; offset += width + 1) {
        const size_t m     = std::min(width, length - offset);
        const double reach = offset == 0 ? 1.0 : ps[offset - 1];
        double sum = 0.0;
        for (size_t j = offset; j < offset + m; ++j) {
            sum += ps[j];
        }
        cost_sum += reach*cost_at(c, m + 1);
        out_sum  += reach + sum;
    }
    return cost_sum/out_sum;
}

size_t common_ngram_varlen_hazard::select(const std::vector<common_ngram_varlen_cand> & alt, int64_t clock, size_t & m,
        const std::vector<double> * cost_in, int32_t width_in) const {
    m = 0;

    const std::vector<double> & c = cost_in != nullptr && !cost_in->empty() ? *cost_in : params.cost;
    const size_t width = (size_t) (width_in > 0 ? width_in : params.width);

    for (size_t ci = 0; ci < alt.size(); ++ci) {
        const auto & cand = alt[ci];
        const size_t len  = cand.tokens.size();

        std::vector<double> ps(len);
        double cumulative = 1.0;
        double last       = 1.0;
        for (size_t j = 1; j <= len; ++j) {
            const feat f = { cand.n, cand.raw[j - 1], cand.support[j - 1] };

            double s, c;
            read(global, key_global((int32_t) j, f.support), clock, s, c);
            const double prior = (s + 1.0)/(c + 2.0);
            read(cells, key_cell((int32_t) j, f), clock, s, c);
            const double q = (s + params.prior_strength*prior)/(c + params.prior_strength);

            cumulative *= q;
            last = std::min(last, std::max(0.001, std::min(0.999, cumulative)));
            ps[j - 1] = last;
        }

        double best = cost_at(c, 1);
        for (size_t j = 1; j <= len; ++j) {
            const double value = expected(ps, j, c, width);
            if (value < best - 1e-10) {
                best = value;
                m    = j;
            }
        }
        if (m > 0) {
            return ci;
        }
    }

    return 0;
}

void common_ngram_varlen_hazard::start(track & t, std::vector<feat> feats, int64_t clock) {
    t.feats   = std::move(feats);
    t.clock0  = clock;
    t.outputs = 0;
    t.offset  = 0;
    t.live    = false;
    t.used    = !t.feats.empty();
}

void common_ngram_varlen_hazard::finish(track & t, int32_t accepted) {
    const int64_t stamp = t.clock0 + t.outputs;
    for (int32_t j = 1; j <= (int32_t) t.feats.size() && j <= accepted + 1; ++j) {
        add(t.feats[j - 1], j, accepted >= j ? 1 : 0, 1.0, stamp);
    }
    drop(t);
}

void common_ngram_varlen_hazard::piece(track & t, size_t m_emitted, size_t n_accepted) {
    if (!t.used) {
        return;
    }

    t.outputs += (int64_t) n_accepted + 1;
    t.live = false;
    if (n_accepted < m_emitted) {
        finish(t, t.offset + (int32_t) n_accepted);
        return;
    }

    t.offset += (int32_t) m_emitted;
    if (t.offset >= (int32_t) t.feats.size()) {
        finish(t, (int32_t) t.feats.size());
        return;
    }
    t.live = true;
}

void common_ngram_varlen_hazard::gate(track & t, bool match) {
    if (!t.used || !t.live) {
        return;
    }

    t.live = false;
    if (!match) {
        finish(t, t.offset);
        return;
    }

    t.offset += 1;
    if (t.offset >= (int32_t) t.feats.size()) {
        finish(t, (int32_t) t.feats.size());
    }
}

void common_ngram_varlen_hazard::drop(track & t) {
    t.feats.clear();
    t.used = false;
    t.live = false;
}
