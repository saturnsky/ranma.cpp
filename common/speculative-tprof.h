#pragma once

// ranma: the trigram confidence of --spec-smart (docs/ranma/spec-smart.md, "Trigram confidence"):
// - the role of every token of a request (assistant or other), from the role markers of the chat template.
// This file has no llama dependency, so that it can be tested on the CPU.

#include <cstddef>
#include <cstdint>
#include <vector>

// the role of every token of a stream. A marker is the token sequence that opens a message of a role (from the chat
// template, e.g. <|assistant|>); a control token that ends no marker (an end-of-turn token, a tool response marker)
// opens a message of another role. Without markers the prompt is OTHER and the generated tokens are ASSIST.
struct common_spec_roles {
    static constexpr uint8_t OTHER  = 0;
    static constexpr uint8_t ASSIST = 1;

    struct marker {
        std::vector<int32_t> toks;
        uint8_t              role = OTHER;
    };
    std::vector<marker>  markers;
    std::vector<uint8_t> ctrl; // per token: a control token that is not a reasoning tag

    bool on() const { return !markers.empty(); }

    // the role of token i of the stream at(0..i], given the role of token i - 1
    template <typename F>
    uint8_t step(uint8_t role, const F & at, size_t i) const {
        const int32_t t = at(i);
        for (const auto & m : markers) {
            const size_t L = m.toks.size();
            if (L == 0 || L > i + 1 || m.toks[L - 1] != t) {
                continue;
            }
            bool eq = true;
            for (size_t j = 0; j + 1 < L && eq; ++j) {
                eq = at(i + 1 - L + j) == m.toks[j];
            }
            if (eq) {
                return m.role;
            }
        }
        if (t >= 0 && (size_t) t < ctrl.size() && ctrl[t]) {
            return OTHER;
        }
        return role;
    }

    // the role of every token of toks[0..n): with markers from the markers, the tokens from gen_from on ASSIST
    std::vector<uint8_t> of(const int32_t * toks, size_t n, size_t gen_from) const {
        std::vector<uint8_t> r(n, OTHER);
        uint8_t cur = OTHER;
        auto at = [&](size_t i) { return toks[i]; };
        for (size_t i = 0; i < n; ++i) {
            if (on()) {
                cur = step(cur, at, i);
            }
            r[i] = i >= gen_from ? ASSIST : cur;
        }
        return r;
    }
};
