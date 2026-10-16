#include "ngram-varlen.h"

#include <algorithm>
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

common_ngram_varlen_draft common_ngram_varlen::draft(llama_seq_id seq_id) const {
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

    std::vector<int32_t> active;
    std::vector<int32_t> keep;
    std::vector<std::pair<llama_token, double>> counts;

    for (const int32_t n : ns) {
        if (n < params.n_min) {
            break;
        }

        // the cohort: every sample that matches at least n tokens, in sample order
        active.clear();
        for (int32_t k = 0; k < (int32_t) matches.size(); ++k) {
            if (matches[k].n >= n) {
                active.push_back(k);
            }
        }
        const int32_t n_cohort = (int32_t) active.size();
        if ((size_t) n_cohort < n_support) {
            continue;
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

        std::vector<llama_token> path;
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
            if (prob + 1e-12 < params.p_min) {
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
            if (support < n_support || (int64_t) j + 1 > (int64_t) params.ladder*(int64_t) (support - 1)) {
                break;
            }
            path.push_back(token);
        }

        if (!path.empty()) {
            res.tokens    = std::move(path);
            res.n         = n;
            res.n_samples = n_cohort;
            return res;
        }
    }

    return res;
}
