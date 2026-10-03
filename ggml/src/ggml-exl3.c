// EXL3 trellis weights (mul1, mcg and 3inst codebooks)
// format and codebooks: ExLlamaV3, Copyright (c) 2025 Turboderp, MIT license
//
// A weight is ne = [K, N(, E)] with ne0 = input. EXL3 stores 16x16 tiles of 256 states each.
// The bytes of a tile are the original bytes. Only the tile order changes: the N rows are split in
// groups of 128, and a group stores [K/16][8 n-tiles][tile]. So one group is 128*K*bits/8 contiguous
// bytes, and nb[1] = K*bits/8, nb[2] = N*nb[1] follow the usual ggml rules at 128-row boundaries.
//
// A tile is a tail-biting ring of 256 positions in a stream of 32-bit little-endian words, read
// MSB first. Position p ends at bit ((p + 1)*bits2) >> 1, bits2 = 2*bits: integer rates use bits per
// position, half rates alternate floor(bits) (even p) and floor(bits) + 1 (odd p). The state of p
// is the 16-bit window that ends there, wrapped around the ring. The codebook of the type maps a state to
// its fp16 value; half rates exist only with mul1.

#include "ggml-exl3.h"
#include "ggml-impl.h"

#include <stdint.h>
#include <string.h>

bool ggml_is_exl3(enum ggml_type type) {
    return (type >= GGML_TYPE_EXL3_M1 && type <= GGML_TYPE_EXL3_M3H) ||
           (type >= GGML_TYPE_EXL3_G1 && type <= GGML_TYPE_EXL3_G8) ||
           (type >= GGML_TYPE_EXL3_T1 && type <= GGML_TYPE_EXL3_T8);
}

// twice the bits per weight
static int ggml_exl3_bits2(enum ggml_type type) {
    const int code = ((int) type - GGML_TYPE_EXL3_M1) % 16;
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

// mcg and 3inst: the hashed state as two fp16 halves (sign and low bits kept, exponent 12..15), added in fp16
// (exact in fp32, then rounded once)
static ggml_fp16_t ggml_exl3_half_sum(uint32_t x) {
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return GGML_FP32_TO_FP16(GGML_FP16_TO_FP32((ggml_fp16_t) (x & 0xffff)) + GGML_FP16_TO_FP32((ggml_fp16_t) (x >> 16)));
}

static ggml_fp16_t ggml_exl3_mcg(uint32_t state) {
    return ggml_exl3_half_sum(state * 0xCBAC1FEDu);
}

static ggml_fp16_t ggml_exl3_3inst(uint32_t state) {
    return ggml_exl3_half_sum(state * 89226354u + 64248484u);
}

// position of tile element (r = row in K, c = column in N)
static int ggml_exl3_position(int r, int c) {
    return ((c % 8)*4 + (r % 8)/2)*8 + r % 2 + (r >= 8 ? 2 : 0) + (c >= 8 ? 4 : 0);
}

void ggml_exl3_decode_group(enum ggml_type type, const void * group, int64_t k, ggml_fp16_t * dst) {
    GGML_ASSERT(ggml_is_exl3(type));
    GGML_ASSERT(k > 0 && k % 16 == 0);

    const int bits2      = ggml_exl3_bits2(type);
    const int codebook   = ((int) type - GGML_TYPE_EXL3_M1) / 16;
    ggml_fp16_t (*const decode)(uint32_t) = codebook == 0 ? ggml_exl3_mul1 : codebook == 1 ? ggml_exl3_mcg : ggml_exl3_3inst;
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
                values[p] = decode((uint32_t) (window >> (48 - start % 32)) & 0xffff);
            }
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    dst[(nt*16 + c)*k + kt*16 + r] = values[position[r*16 + c]];
                }
            }
        }
    }
}

// EXL3 row codec (ExLlamaV3 exl3_ngram_trellis, modules/quant/exl3_lib/ngram_codec.py)
//
// A row of 160 values is a tail-biting ring over the mul1 codebook: an fp16 scale (2 bytes), then 160*bits bits in
// little-endian 16-bit words, read LSB first. Position i owns stream bits [i*bits, (i+1)*bits), and its state is the
// 16 ring bits that end there: its own bits low, then those of i-1, i-2, ... (mod 160). Each row decodes alone.
// The value is the fp16 codebook value times the scale, in f32 (both are fp16, so the product is exact). The official
// reconstruction then adds a per-head bias and rounds to fp16; that is the graph's job, not the row's.

bool ggml_is_exl3_row(enum ggml_type type) {
    return type >= GGML_TYPE_EXL3R_M1 && type <= GGML_TYPE_EXL3R_M8;
}

static void ggml_exl3r_dequantize(const uint8_t * x, float * y, int64_t k, int bits) {
    enum { ROW = 160 };
    GGML_ASSERT(k % ROW == 0);
    const int64_t row_bytes = 2 + ROW*bits/8;
    const uint32_t mask = (1u << bits) - 1;

    for (int64_t r = 0; r < k/ROW; ++r) {
        const uint8_t * src = x + r*row_bytes;
        float         * dst = y + r*ROW;

        const float scale = GGML_FP16_TO_FP32((ggml_fp16_t) (src[0] | (src[1] << 8)));
        const uint8_t * stream = src + 2;

        uint32_t code[ROW];
        for (int i = 0; i < ROW; ++i) {
            const int b0 = i*bits;
            // the bits of position i span at most 2 bytes (bits <= 8); the last position has no byte after it
            uint32_t w = stream[b0 >> 3];
            if ((b0 >> 3) + 1 < ROW*bits/8) {
                w |= (uint32_t) stream[(b0 >> 3) + 1] << 8;
            }
            code[i] = (w >> (b0 & 7)) & mask;
        }
        for (int i = 0; i < ROW; ++i) {
            uint32_t state = 0;
            for (int j = 0; j*bits < 16; ++j) {
                state |= code[(i - j + ROW) % ROW] << (j*bits);
            }
            dst[i] = GGML_FP16_TO_FP32(ggml_exl3_mul1(state & 0xffff)) * scale;
        }
    }
}

#define GGML_EXL3R_DEQUANTIZE(bits) \
    void dequantize_row_exl3r_m##bits(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) { \
        ggml_exl3r_dequantize((const uint8_t *) x, y, k, bits); \
    }

GGML_EXL3R_DEQUANTIZE(1)
GGML_EXL3R_DEQUANTIZE(2)
GGML_EXL3R_DEQUANTIZE(3)
GGML_EXL3R_DEQUANTIZE(4)
GGML_EXL3R_DEQUANTIZE(5)
GGML_EXL3R_DEQUANTIZE(6)
GGML_EXL3R_DEQUANTIZE(7)
GGML_EXL3R_DEQUANTIZE(8)
