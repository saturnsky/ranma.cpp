#pragma once

#include <cstdint>
#include <cstring>

// PLE n-gram hash helpers, shared by the graph input and by standalone checks.

// context[0] = current, context[s] = the token s positions back.
// previous[] is oldest-first; a negative entry is a missing predecessor.
// An EOS or a missing predecessor cuts the window: it and everything before it read as EOS.
// The EOS of the current token does not cut its own context.
inline void llama_ple_context(int32_t current, const int32_t * previous, int64_t n_prev,
        int64_t eos, int64_t * context) {
    context[0] = current;
    bool cut = false;
    for (int64_t s = 1; s <= n_prev; ++s) {
        const int32_t token = cut ? -1 : previous[n_prev - s];
        cut = cut || token < 0 || token == eos;
        context[s] = cut ? eos : token;
    }
}

//   mixed_n = (t[0]*m[0]) ^ ... ^ (t[n-1]*m[n-1]);  row = mixed_n mod vocab[h] + offset[h]
// The products and the xor wrap in 64 bits. The reference hashes in int64, so the wrapped bits are read
// as a signed value and reduced with a non-negative remainder. For the shipped multipliers and vocabularies
// the products stay below 2^63 and this equals the unsigned remainder; it only differs past that.
inline void llama_ple_hash(const int64_t * context, const uint64_t * multipliers, int64_t n_gram,
        int64_t per_gram, const uint32_t * offsets, const uint32_t * sizes, int32_t * output) {
    uint64_t mixed = uint64_t(context[0]) * multipliers[0];
    for (int64_t n = 2; n <= n_gram; ++n) {
        mixed ^= uint64_t(context[n - 1]) * multipliers[n - 1];
        int64_t signed_mixed;
        memcpy(&signed_mixed, &mixed, sizeof(mixed));
        for (int64_t g = 0; g < per_gram; ++g) {
            const int64_t head = (n - 2) * per_gram + g;
            int64_t row = signed_mixed % int64_t(sizes[head]);
            if (row < 0) {
                row += sizes[head];
            }
            output[head] = int32_t(row + offsets[head]);
        }
    }
}
