// EXL3 trellis weights (mul1 codebook)
// format and codebook: ExLlamaV3, Copyright (c) 2025 Turboderp, MIT license
//
// A weight is ne = [K, N(, E)] with ne0 = input. EXL3 stores 16x16 tiles of 256 states each.
// The bytes of a tile are the original bytes. Only the tile order changes: the N rows are split in
// groups of 128, and a group stores [K/16][8 n-tiles][tile]. So one group is 128*K*bits/8 contiguous
// bytes, and nb[1] = K*bits/8, nb[2] = N*nb[1] follow the usual ggml rules at 128-row boundaries.
//
// A tile is a tail-biting ring of 256 positions in a stream of 32-bit little-endian words, read
// MSB first. Position p ends at bit ((p + 1)*bits2) >> 1, bits2 = 2*bits: integer rates use bits per
// position, half rates alternate floor(bits) (even p) and floor(bits) + 1 (odd p). The state of p
// is the 16-bit window that ends there, wrapped around the ring.

#include "ggml-impl.h"

#include <stdint.h>
#include <string.h>

bool ggml_is_exl3(enum ggml_type type) {
    return type >= GGML_TYPE_EXL3_M1 && type <= GGML_TYPE_EXL3_M3H;
}

// twice the bits per weight
static int ggml_exl3_bits2(enum ggml_type type) {
    const int code = (int) type - GGML_TYPE_EXL3_M1;
    return code < 8 ? 2*(code + 1) : 2*(code - 7) + 1;
}

static uint32_t ggml_exl3_word(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

// mul1 codebook: byte sum of the hashed state, then one fp16 FMA (exact in fp32, then rounded once)
static ggml_fp16_t ggml_exl3_mul1(uint32_t state) {
    const uint32_t x = state * 0x83DCD12Du;
    const uint32_t sum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
    const float k_inv  = GGML_FP16_TO_FP32(0x1eee);
    const float k_bias = GGML_FP16_TO_FP32(0xc931);
    return GGML_FP32_TO_FP16((float) (1024 + sum) * k_inv + k_bias);
}

// position of tile element (r = row in K, c = column in N)
static int ggml_exl3_position(int r, int c) {
    return ((c % 8)*4 + (r % 8)/2)*8 + r % 2 + (r >= 8 ? 2 : 0) + (c >= 8 ? 4 : 0);
}

void ggml_exl3_decode_group(enum ggml_type type, const void * group, int64_t k, ggml_fp16_t * dst) {
    GGML_ASSERT(ggml_is_exl3(type));
    GGML_ASSERT(k > 0 && k % 16 == 0);

    const int bits2      = ggml_exl3_bits2(type);
    const int ring_bits  = 128*bits2;
    const int n_words    = ring_bits/32;
    const int tile_bytes = 16*bits2;

    int position[256];
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            position[r*16 + c] = ggml_exl3_position(r, c);
        }
    }

    const uint8_t * src = (const uint8_t *) group;
    for (int64_t kt = 0; kt < k/16; ++kt) {
        for (int nt = 0; nt < 8; ++nt) {
            const uint8_t * tile = src + (kt*8 + nt)*tile_bytes;
            ggml_fp16_t values[256];
            for (int p = 0; p < 256; ++p) {
                const int end   = ((p + 1)*bits2) >> 1;
                const int start = (end - 16 + ring_bits) % ring_bits;
                const int w     = start / 32;
                const uint64_t window = ((uint64_t) ggml_exl3_word(tile + 4*w) << 32) | ggml_exl3_word(tile + 4*((w + 1) % n_words));
                values[p] = ggml_exl3_mul1((uint32_t) (window >> (48 - start % 32)) & 0xffff);
            }
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    dst[(nt*16 + c)*k + kt*16 + r] = values[position[r*16 + c]];
                }
            }
        }
    }
}
