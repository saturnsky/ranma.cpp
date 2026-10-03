#pragma once

// EXL3 internals of ggml-base (ggml-exl3.c)

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// to_float of the EXL3 row codec types (GGML_TYPE_EXL3R_M1 .. M8)
void dequantize_row_exl3r_m1(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_exl3r_m2(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_exl3r_m3(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_exl3r_m4(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_exl3r_m5(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_exl3r_m6(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_exl3r_m7(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_exl3r_m8(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif
