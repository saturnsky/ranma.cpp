#include "ggml.h"
#include "common.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "mapped-host.cuh"
#include "convert.cuh"

template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false, bool is_multi_token_id = false>
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const int ncols2, const uint3 nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride) {
    const T       * GGML_CUDA_RESTRICT x   = x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int row         = blockIdx.x;
    // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens)
    const int channel_dst = blockIdx.y;
    const int tid         = threadIdx.x;

    int token_idx;
    int channel_x;
    int channel_y;
    int sample_dst;

    ggml_cuda_pdl_sync();
    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        token_idx  = ids ? blockIdx.z                                          : 0;
        channel_x  = ids ? ids[blockIdx.y + token_idx * ids_stride]            : fastdiv((uint32_t) channel_dst, channel_ratio);
        channel_y  = ids ? fastmodulo(blockIdx.y, nchannels_y)                 : channel_dst;
        sample_dst = ids ? 0                                                   : blockIdx.z;
    }

    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y    = sample_dst;

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if constexpr (is_multi_token_id) {
        y   += token_idx*stride_col_y2*2;
        dst += token_idx*stride_col_dst;
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    ggml_glu_op glu_op = ggml_glu_op::GGML_GLU_OP_SWIGLU;
    float glu_limit = 0.0f;
    const T * gate_x = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr;
        glu_op = fusion.glu_op;
        glu_limit = fusion.glu_limit;

        if (use_gate) {
            gate_x = static_cast<const T *>(fusion.gate);
        }
        if (use_bias) {
            x_bias = static_cast<const float *>(fusion.x_bias);
        }
        if (use_gate_bias) {
            gate_bias = static_cast<const float *>(fusion.gate_bias);
            use_gate_bias = use_gate;
        } else {
            use_gate_bias = false;
        }
    }

    if (use_gate) {
        gate_x += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    }

    if constexpr (has_fusion) {
        const int channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
        if (use_gate_bias) {
            gate_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;
    [[maybe_unused]] float * buf_iw_gate = nullptr;
    if constexpr (has_fusion) {
        buf_iw_gate = (float *) (data_mmv + warp_size*sizeof(float));
    }

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};
    float sumf_gate[ncols_dst];
    if constexpr (has_fusion) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf_gate[j] = 0.0f;
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
        [[maybe_unused]] const float2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const float2 *) gate_x;
            }
        }

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            float2 tmpx_gate = make_float2(0.0f, 0.0f);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2 = (const half2 *) x;
        [[maybe_unused]] const half2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const half2 *) gate_x;
            }
        }

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                float2 tmpx_gate = make_float2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = __half22float2(gate_x2[col2]);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                        }
                    }
                }
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst] = {{0.0f, 0.0f}};
            half2 sumh2_gate[ncols_dst] = {{0.0f, 0.0f}};

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const half2 tmpx = x2[col2];
                half2 tmpx_gate = make_half2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[col2];
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumh2_gate[j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }

            if constexpr (has_fusion) {
                if (use_gate) {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        sumf_gate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
//TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
        const int * x2 = (const int *) x;
        const int * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const int *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const int tmpx = x2[col2];
            int tmpx_gate = 0;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                ggml_cuda_mad(sumf[j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx1, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[0]);
                        const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[1]);
                        ggml_cuda_mad(sumf_gate[j], tmpx0_gate, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx1_gate, tmpy.y);
                    }
                }
            }
        }
#else
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
        [[maybe_unused]] const nv_bfloat162 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const nv_bfloat162 *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const nv_bfloat162 tmpx = x2[col2];
            [[maybe_unused]] nv_bfloat162 tmpx_gate;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
#endif
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

    ggml_cuda_pdl_lc();
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if constexpr (has_fusion) {
            if (use_gate) {
                sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
            }
        }

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid/warp_size] = sumf_gate[j];
                }
            }
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        sumf_gate[j] = buf_iw_gate[tid];
                        sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
                    }
                }
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    float value = sumf[tid];

    if constexpr (has_fusion) {
        if (use_bias) {
            value += x_bias[tid*stride_col_dst + row];
        }

        if (use_gate) {
            float gate_value = sumf_gate[tid];
            if (use_gate_bias) {
                gate_value += gate_bias[tid*stride_col_dst + row];
            }
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    value *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    value *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI: {
                    value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                    break;
                }
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    value = ggml_cuda_op_swiglu_clamp_single(gate_value, value, glu_limit);
                    break;
                default:
                    break;
            }
        }
    }

    dst[tid*stride_col_dst + row] = value;

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, glu_op, glu_limit, gate_x, x_bias, gate_bias, sumf_gate);
    }
}

template<typename T, typename type_acc, int ncols_dst, int block_size, bool is_multi_token_id = false>
static void mul_mat_vec_f_switch_fusion(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const uint3 nchannels_y,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const dim3 & block_dims, const dim3 & block_nums, const int nbytes_shared, const int ids_stride, const cudaStream_t stream) {

    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, nbytes_shared, stream};

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (ncols_dst == 1) {
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, is_multi_token_id>, launch_params,
                x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
       }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, is_multi_token_id>, launch_params,
        x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);

}

// the fewest iterations over a row of ncols values, and the smallest block that reaches them
static int64_t mul_mat_vec_f_block_size(const int64_t ncols) {
    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return block_size_best;
}

template <typename T, typename type_acc, int ncols_dst, bool is_multi_token_id = false>
void launch_mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t nsamples_or_ntokens, const int64_t ids_stride, cudaStream_t stream) {
    GGML_ASSERT(ncols        % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int64_t block_size_best = mul_mat_vec_f_block_size(ncols);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

    const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? warp_size*sizeof(float) : 0);
    const dim3 block_nums(nrows, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case   32: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   64: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 64, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   96: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 96, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  128: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 128, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  160: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 160, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  192: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 192, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  224: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 224, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  256: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, cudaStream_t stream) {

    const bool has_ids = ids != nullptr;

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst, true>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    if (has_ids) {
        // Single-token MUL_MAT_ID path
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_f_cuda<T, type_acc, 1>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 2:
            launch_mul_mat_vec_f_cuda<T, type_acc, 2>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 3:
            launch_mul_mat_vec_f_cuda<T, type_acc, 3>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 4:
            launch_mul_mat_vec_f_cuda<T, type_acc, 4>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 5:
            launch_mul_mat_vec_f_cuda<T, type_acc, 5>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 6:
            launch_mul_mat_vec_f_cuda<T, type_acc, 6>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 7:
            launch_mul_mat_vec_f_cuda<T, type_acc, 7>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 8:
            launch_mul_mat_vec_f_cuda<T, type_acc, 8>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template<typename T>
static void mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream) {

    if constexpr(std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {
    ggml_cuda_assert_src0_is_device_readable(src0);
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(!ids || ne12 <= MMVF_MAX_BATCH_SIZE);
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0  =  dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t stride_col_y       = ne10;
    const int64_t stride_col_dst     = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    ggml_cuda_mm_fusion_args_device empty{};
    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED_VARS(ctx, src1, dst, src1_ddq_i, src1_ncols, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                        return ne11 <= 3;
                    }
                    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        default:
            return false;
    }
}

// Skinny dense F32 x F32 matrix multiplication, dst[j][i] = sum_k x[i][k]*y[j][k], for few weight rows (x) and more
// columns (y) than MMVF takes. RDNA4 has no F32 MMF path, so such products (small projections of a prompt batch,
// e.g. 4 x 512 x 10240 or 48 x 512 x 2560) went to hipBLAS. hipBLAS/hipBLASLt chooses its solution per process,
// so both the speed and the summation order of these products varied from one process to the next. The kernels
// below have a fixed summation order and use no atomics, so a result is the same in every run and process.

// Very few output rows: every wave streams y rows once and keeps all rows of x in registers per k step.
// The K dimension is split over ksplit waves of a block and the partial sums are added in a fixed order.
template <int nrows_x, int ntok_per_wave, int ksplit>
static __global__ void __launch_bounds__(256) mul_mat_f32_skinny_rows(
        const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int nrows, const int ntok, const int ncols4,
        const int64_t stride_x, const int64_t stride_y, const int64_t stride_dst) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwaves    = 256/warp_size;
    constexpr int ngroups   = nwaves/ksplit;
    static_assert(ngroups*ksplit == nwaves, "bad ksplit");
    static_assert(ntok_per_wave*nrows_x <= warp_size, "one lane per output of a wave");

    const int lane = threadIdx.x % warp_size;
    const int wave = threadIdx.x / warp_size;
    const int tg   = wave % ngroups;
    const int ks   = wave / ngroups;
    const int tok0 = (blockIdx.x*ngroups + tg)*ntok_per_wave;

    const float4 * xr[nrows_x];
#pragma unroll
    for (int i = 0; i < nrows_x; ++i) {
        xr[i] = (const float4 *) (x + min(i, nrows - 1)*stride_x);
    }
    const float4 * yr[ntok_per_wave];
#pragma unroll
    for (int t = 0; t < ntok_per_wave; ++t) {
        yr[t] = (const float4 *) (y + min(tok0 + t, ntok - 1)*stride_y);
    }

    float sum[ntok_per_wave][nrows_x] = {{0.0f}};

#pragma unroll 2
    for (int c = ks*warp_size + lane; c < ncols4; c += ksplit*warp_size) {
        float4 yv[ntok_per_wave];
#pragma unroll
        for (int t = 0; t < ntok_per_wave; ++t) {
            yv[t] = yr[t][c];
        }
#pragma unroll
        for (int i = 0; i < nrows_x; ++i) {
            const float4 xv = xr[i][c];
#pragma unroll
            for (int t = 0; t < ntok_per_wave; ++t) {
                ggml_cuda_mad(sum[t][i], xv.x, yv[t].x);
                ggml_cuda_mad(sum[t][i], xv.y, yv[t].y);
                ggml_cuda_mad(sum[t][i], xv.z, yv[t].z);
                ggml_cuda_mad(sum[t][i], xv.w, yv[t].w);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < ntok_per_wave; ++t) {
#pragma unroll
        for (int i = 0; i < nrows_x; ++i) {
            sum[t][i] = warp_reduce_sum<warp_size>(sum[t][i]);
        }
    }

    if constexpr (ksplit == 1) {
#pragma unroll
        for (int t = 0; t < ntok_per_wave; ++t) {
#pragma unroll
            for (int i = 0; i < nrows_x; ++i) {
                if (lane == t*nrows_x + i && tok0 + t < ntok && i < nrows) {
                    dst[(tok0 + t)*stride_dst + i] = sum[t][i];
                }
            }
        }
    } else {
        __shared__ float partial[ksplit][ngroups][ntok_per_wave][nrows_x];
#pragma unroll
        for (int t = 0; t < ntok_per_wave; ++t) {
#pragma unroll
            for (int i = 0; i < nrows_x; ++i) {
                if (lane == t*nrows_x + i) {
                    partial[ks][tg][t][i] = sum[t][i];
                }
            }
        }
        __syncthreads();

        constexpr int nout = ngroups*ntok_per_wave*nrows_x;
        for (int idx = threadIdx.x; idx < nout; idx += 256) {
            const int i   = idx % nrows_x;
            const int t   = (idx / nrows_x) % ntok_per_wave;
            const int g   = idx / (nrows_x*ntok_per_wave);
            const int tok = (blockIdx.x*ngroups + g)*ntok_per_wave + t;
            float acc = partial[0][g][t][i];
#pragma unroll
            for (int s = 1; s < ksplit; ++s) {
                acc += partial[s][g][t][i];
            }
            if (tok < ntok && i < nrows) {
                dst[tok*stride_dst + i] = acc;
            }
        }
    }
}

// More output rows: register-tiled GEMM with both operands staged through shared memory, K split over ksplit
// thread groups of a block and optionally over gridDim.z blocks. The latter write partial sums that
// mul_mat_f32_skinny_reduce adds in a fixed order.
template <int bm, int bn, int bk, int tm, int tn, int ksplit>
static __global__ void __launch_bounds__((bm/tm)*(bn/tn)*ksplit) mul_mat_f32_skinny_tile(
        const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int nrows, const int ntok, const int ncols, const int ncols_chunk,
        const int64_t stride_x, const int64_t stride_y, const int64_t stride_dst, const int64_t stride_part) {
    constexpr int ntg      = (bm/tm)*(bn/tn);
    constexpr int nthreads = ntg*ksplit;
    constexpr int bkt      = bk*ksplit;
    constexpr int bkt4     = bkt/4;
    constexpr int na4      = bm*bkt4;
    constexpr int nb4      = bn*bkt4;
    constexpr int la       = (na4 + nthreads - 1)/nthreads;
    constexpr int lb       = (nb4 + nthreads - 1)/nthreads;
    constexpr int pad      = 4;
    // Each thread owns tm/4 groups of 4 rows and tn/vn groups of vn tokens, the groups spaced so that
    // consecutive threads read consecutive float4 from shared memory.
    constexpr int vn       = tn % 4 == 0 ? 4 : tn;
    constexpr int sm       = (bm/tm)*4;
    constexpr int sn       = (bn/tn)*vn;
    static_assert(bk % 4 == 0 && tm % 4 == 0 && tn % vn == 0, "bad tile");

    // Rows are read as float4, so the tiles are 16-byte aligned and the padded row length is a multiple of 4.
    __shared__ __align__(16) float xs[bkt][bm + pad];
    __shared__ __align__(16) float ys[bkt][bn + pad];

    const int tid = threadIdx.x;
    const int g   = tid / ntg;
    const int lt  = tid % ntg;
    const int tx  = lt % (bn/tn);
    const int ty  = lt / (bn/tn);

    const int i0   = blockIdx.y*bm;
    const int j0   = blockIdx.x*bn;
    const int kbeg = blockIdx.z*ncols_chunk;
    const int kend = min(ncols, kbeg + ncols_chunk);
    dst += blockIdx.z*stride_part;

    float4 rx[la];
    float4 ry[lb];

    auto load = [&](const int kb) {
#pragma unroll
        for (int l = 0; l < la; ++l) {
            const int f  = tid + l*nthreads;
            const int r  = f / bkt4;
            const int kk = kb + (f % bkt4)*4;
            rx[l] = f < na4 && i0 + r < nrows && kk < kend ?
                *(const float4 *) (x + (i0 + r)*stride_x + kk) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int l = 0; l < lb; ++l) {
            const int f  = tid + l*nthreads;
            const int r  = f / bkt4;
            const int kk = kb + (f % bkt4)*4;
            ry[l] = f < nb4 && j0 + r < ntok && kk < kend ?
                *(const float4 *) (y + (j0 + r)*stride_y + kk) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };

    float sum[tn][tm] = {{0.0f}};

    load(kbeg);
    for (int kb = kbeg; kb < kend; kb += bkt) {
#pragma unroll
        for (int l = 0; l < la; ++l) {
            const int f = tid + l*nthreads;
            if (f < na4) {
                const int r = f / bkt4;
                const int k = (f % bkt4)*4;
                xs[k + 0][r] = rx[l].x;
                xs[k + 1][r] = rx[l].y;
                xs[k + 2][r] = rx[l].z;
                xs[k + 3][r] = rx[l].w;
            }
        }
#pragma unroll
        for (int l = 0; l < lb; ++l) {
            const int f = tid + l*nthreads;
            if (f < nb4) {
                const int r = f / bkt4;
                const int k = (f % bkt4)*4;
                ys[k + 0][r] = ry[l].x;
                ys[k + 1][r] = ry[l].y;
                ys[k + 2][r] = ry[l].z;
                ys[k + 3][r] = ry[l].w;
            }
        }
        __syncthreads();

        if (kb + bkt < kend) {
            load(kb + bkt);
        }

#pragma unroll
        for (int kk = 0; kk < bk; ++kk) {
            const int k = g*bk + kk;
            float xv[tm];
            float yv[tn];
#pragma unroll
            for (int a = 0; a < tm; a += 4) {
                const float4 v = *(const float4 *) &xs[k][(a/4)*sm + ty*4];
                xv[a + 0] = v.x;
                xv[a + 1] = v.y;
                xv[a + 2] = v.z;
                xv[a + 3] = v.w;
            }
            if constexpr (vn == 4) {
#pragma unroll
                for (int b = 0; b < tn; b += 4) {
                    const float4 v = *(const float4 *) &ys[k][(b/4)*sn + tx*4];
                    yv[b + 0] = v.x;
                    yv[b + 1] = v.y;
                    yv[b + 2] = v.z;
                    yv[b + 3] = v.w;
                }
            } else {
#pragma unroll
                for (int b = 0; b < tn; ++b) {
                    yv[b] = ys[k][tx*tn + b];
                }
            }
#pragma unroll
            for (int b = 0; b < tn; ++b) {
#pragma unroll
                for (int a = 0; a < tm; ++a) {
                    ggml_cuda_mad(sum[b][a], xv[a], yv[b]);
                }
            }
        }
        __syncthreads();
    }

    if constexpr (ksplit > 1) {
        __shared__ float partial[ksplit - 1][ntg][tn][tm];
        if (g > 0) {
#pragma unroll
            for (int b = 0; b < tn; ++b) {
#pragma unroll
                for (int a = 0; a < tm; ++a) {
                    partial[g - 1][lt][b][a] = sum[b][a];
                }
            }
        }
        __syncthreads();
        if (g > 0) {
            return;
        }
#pragma unroll
        for (int s = 0; s < ksplit - 1; ++s) {
#pragma unroll
            for (int b = 0; b < tn; ++b) {
#pragma unroll
                for (int a = 0; a < tm; ++a) {
                    sum[b][a] += partial[s][lt][b][a];
                }
            }
        }
    }

#pragma unroll
    for (int b = 0; b < tn; ++b) {
        const int j = j0 + (b/vn)*sn + tx*vn + b%vn;
        if (j >= ntok) {
            continue;
        }
#pragma unroll
        for (int a = 0; a < tm; ++a) {
            const int i = i0 + (a/4)*sm + ty*4 + a%4;
            if (i < nrows) {
                dst[j*stride_dst + i] = sum[b][a];
            }
        }
    }
}

static __global__ void mul_mat_f32_skinny_reduce(
        const float * __restrict__ part, float * __restrict__ dst, const int nrows, const int ntok, const int nparts,
        const int64_t stride_dst) {
    const int64_t n   = int64_t(nrows)*ntok;
    const int64_t idx = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float acc = part[idx];
    for (int p = 1; p < nparts; ++p) {
        acc += part[p*n + idx];
    }
    dst[(idx / nrows)*stride_dst + idx % nrows] = acc;
}

template <int nrows_x, int ntok_per_wave, int ksplit>
static void mul_mat_f32_skinny_rows_cuda(
        const float * x, const float * y, float * dst, const int nrows, const int ntok, const int ncols,
        const int64_t stride_x, const int64_t stride_y, const int64_t stride_dst, cudaStream_t stream) {
    constexpr int warp_size = 32; // the dispatch is limited to RDNA4, which runs these kernels in wave32
    constexpr int ntok_per_block = (256/warp_size/ksplit)*ntok_per_wave;
    const dim3 grid((ntok + ntok_per_block - 1)/ntok_per_block, 1, 1);
    mul_mat_f32_skinny_rows<nrows_x, ntok_per_wave, ksplit><<<grid, 256, 0, stream>>>
        (x, y, dst, nrows, ntok, ncols/4, stride_x, stride_y, stride_dst);
}

template <int bm, int bn, int bk, int tm, int tn, int ksplit>
static void mul_mat_f32_skinny_tile_cuda(
        ggml_backend_cuda_context & ctx, const float * x, const float * y, float * dst, const int nrows, const int ntok,
        const int ncols, const int64_t stride_x, const int64_t stride_y, const int64_t stride_dst, cudaStream_t stream) {
    constexpr int nthreads = (bm/tm)*(bn/tn)*ksplit;
    constexpr int bkt      = bk*ksplit;
    const int nchunks     = (ncols + bkt - 1)/bkt;
    // Split K over the grid until there are about four blocks per SM, keeping at least four shared memory stages per
    // block. A block is latency bound on its serial K loop, so few large blocks leave the GPU mostly idle, while more
    // parts cost partial-sum traffic: allow 16 parts only for tiny grids.
    const int nblocks_mn = ((ntok + bn - 1)/bn)*((nrows + bm - 1)/bm);
    const int nsm        = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int max_parts  = nblocks_mn <= 4 ? 16 : 8;
    const int nparts     = std::max(1, std::min({max_parts, (4*nsm + nblocks_mn - 1)/nblocks_mn, nchunks/4}));
    const int ncols_chunk = ((nchunks + nparts - 1)/nparts)*bkt;
    const dim3 grid((ntok + bn - 1)/bn, (nrows + bm - 1)/bm, nparts);

    if (nparts == 1) {
        mul_mat_f32_skinny_tile<bm, bn, bk, tm, tn, ksplit><<<grid, nthreads, 0, stream>>>
            (x, y, dst, nrows, ntok, ncols, ncols_chunk, stride_x, stride_y, stride_dst, 0);
        return;
    }

    const int64_t n = int64_t(nrows)*ntok;
    ggml_cuda_pool_alloc<float> part(ctx.pool(), nparts*n);
    mul_mat_f32_skinny_tile<bm, bn, bk, tm, tn, ksplit><<<grid, nthreads, 0, stream>>>
        (x, y, part.get(), nrows, ntok, ncols, ncols_chunk, stride_x, stride_y, nrows, n);
    const int nblocks = (n + 255)/256;
    mul_mat_f32_skinny_reduce<<<nblocks, 256, 0, stream>>>(part.get(), dst, nrows, ntok, nparts, stride_dst);
}

bool ggml_cuda_should_use_mul_mat_f32_skinny(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    // GGML_CUDA_SKINNY_F32=0 restores hipBLAS for these products.
    static const bool enabled = [] {
        const char * v = getenv("GGML_CUDA_SKINNY_F32");
        return !v || atoi(v) != 0;
    }();
    if (!enabled || !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->buffer && ggml_backend_buffer_is_host(src0->buffer) && !ggml_cuda_info().devices[ggml_cuda_get_device()].integrated) {
        return false;
    }
    // Plain 2D product with rows that can be read as float4.
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->ne[0] % 4 != 0 || src0->nb[1] % 16 != 0 || src1->nb[1] % 16 != 0 ||
        (uintptr_t) src0->data % 16 != 0 || (uintptr_t) src1->data % 16 != 0) {
        return false;
    }
    const int64_t nrows = src0->ne[1];
    const int64_t ntok  = src1->ne[1];
    if (ntok <= MMVF_MAX_BATCH_SIZE || nrows > 512 || src0->ne[0] > INT_MAX/2 || nrows*ntok > INT_MAX/8) {
        return false;
    }
    return true;
}

void ggml_cuda_mul_mat_f32_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const float * x_d   = (const float *) src0->data;
    const float * y_d   = (const float *) src1->data;
    float       * dst_d = (float       *) dst->data;

    const int     nrows      = src0->ne[1];
    const int     ntok       = src1->ne[1];
    const int     ncols      = src0->ne[0];
    const int64_t stride_x   = src0->nb[1] / sizeof(float);
    const int64_t stride_y   = src1->nb[1] / sizeof(float);
    const int64_t stride_dst = dst->nb[1]  / sizeof(float);
    cudaStream_t  stream     = ctx.stream();

    // The kernel shapes were tuned on RDNA4 for the 4 x n x 10240, 48 x n x 2560 and 512 x n x 2560 products; other
    // shapes within the dispatch bounds take the nearest configuration.
    if (nrows <= 4) {
        mul_mat_f32_skinny_rows_cuda<4, 2, 8>(x_d, y_d, dst_d, nrows, ntok, ncols, stride_x, stride_y, stride_dst, stream);
    } else if (nrows <= 8) {
        mul_mat_f32_skinny_rows_cuda<8, 1, 4>(x_d, y_d, dst_d, nrows, ntok, ncols, stride_x, stride_y, stride_dst, stream);
    } else if (nrows <= 64 && ntok <= 128) {
        mul_mat_f32_skinny_tile_cuda<16, 32, 8, 4, 2, 4>(ctx, x_d, y_d, dst_d, nrows, ntok, ncols, stride_x, stride_y, stride_dst, stream);
    } else if (nrows <= 64) {
        mul_mat_f32_skinny_tile_cuda<48, 32, 8, 4, 4, 4>(ctx, x_d, y_d, dst_d, nrows, ntok, ncols, stride_x, stride_y, stride_dst, stream);
    } else if (ntok <= 32) {
        mul_mat_f32_skinny_tile_cuda<64, 32, 16, 4, 2, 1>(ctx, x_d, y_d, dst_d, nrows, ntok, ncols, stride_x, stride_y, stride_dst, stream);
    } else {
        mul_mat_f32_skinny_tile_cuda<128, 64, 8, 8, 8, 1>(ctx, x_d, y_d, dst_d, nrows, ntok, ncols, stride_x, stride_y, stride_dst, stream);
    }
}

// GGML_OP_RELU_SUM_HEADS: the rectified head sum of an indexer score product, alone or fused with the MMVF product that
// produces it. Both keep the rounding of the op chain they replace (unary relu, cont, add, add, add, add bias):
// op_relu is fmaxf(x, 0), the heads are added left to right and the bias last, and nothing here multiplies, so no
// contraction can change a sum. The fused kernel repeats the F32 path of mul_mat_vec_f step for step.

static __device__ __forceinline__ float relu_sum_heads_relu(const float x) {
    return fmaxf(x, 0);
}

static __global__ void relu_sum_heads_f32(
        const float * __restrict__ a, const float * __restrict__ bias, float * __restrict__ dst,
        const int n, const int n_head, const int n2,
        const int64_t s_a1, const int64_t s_a2, const int64_t s_a3,
        const int64_t s_b1, const int64_t s_b2, const int64_t s_b3,
        const int64_t s_d1, const int64_t s_d2, const int64_t s_d3) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t i1 = blockIdx.y;
    const int64_t i2 = blockIdx.z % n2;
    const int64_t i3 = blockIdx.z / n2;

    const float * a_row = a + i1*n_head*s_a1 + i2*s_a2 + i3*s_a3;

    float acc = relu_sum_heads_relu(a_row[i]);
    for (int h = 1; h < n_head; ++h) {
        acc = acc + relu_sum_heads_relu(a_row[h*s_a1 + i]);
    }
    if (bias) {
        acc = acc + bias[i1*s_b1 + i2*s_b2 + i3*s_b3 + i];
    }
    dst[i1*s_d1 + i2*s_d2 + i3*s_d3 + i] = acc;
}

void ggml_cuda_op_relu_sum_heads(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a    = dst->src[0];
    const ggml_tensor * bias = dst->src[1];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));
    GGML_ASSERT(!bias || (bias->type == GGML_TYPE_F32 && bias->nb[0] == sizeof(float)));

    const int n_head = ggml_get_op_params_i32(dst, 0);

    const int64_t n  = dst->ne[0];
    const int64_t n1 = dst->ne[1];
    const int64_t n2 = dst->ne[2];
    const int64_t n3 = dst->ne[3];
    GGML_ASSERT(n <= INT_MAX && n1 <= 65535 && n2*n3 <= 65535);


    const int64_t ts = sizeof(float);
    const int block_size = 256;
    const dim3 block_nums((n + block_size - 1)/block_size, n1, n2*n3);
    relu_sum_heads_f32<<<block_nums, block_size, 0, ctx.stream()>>>(
        (const float *) a->data, bias ? (const float *) bias->data : nullptr, (float *) dst->data,
        n, n_head, n2,
        a->nb[1]/ts, a->nb[2]/ts, a->nb[3]/ts,
        bias ? bias->nb[1]/ts : 0, bias ? bias->nb[2]/ts : 0, bias ? bias->nb[3]/ts : 0,
        dst->nb[1]/ts, dst->nb[2]/ts, dst->nb[3]/ts);
    CUDA_CHECK(cudaGetLastError());
}

// One block per output row as in mul_mat_vec_f<float, float, ncols_dst, block_size> without ids or fusion, with sample
// and channel ratios of 1. Only thread 0 writes: after the reductions every lane of warp 0 holds the same sums, and
// mul_mat_vec_f writes column j from lane j.
template <int ncols_dst, int block_size, int n_head>
static __global__ void mul_mat_vec_f_relu_sum_heads(
        const float * __restrict__ x, const float * __restrict__ y, const float * __restrict__ bias, float * __restrict__ dst,
        const int ncols2, const int stride_row, const int stride_col_y2,
        const int stride_channel_x, const int stride_channel_y, const int64_t stride_sample_x, const int64_t stride_sample_y,
        const int stride_col_dst, const int stride_channel_dst, const int64_t stride_sample_dst,
        const int stride_col_bias, const int stride_channel_bias, const int64_t stride_sample_bias) {
    static_assert(ncols_dst % n_head == 0, "whole tokens only");
    constexpr int n_tokens  = ncols_dst/n_head;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x += int64_t(sample)*stride_sample_x + channel*stride_channel_x + row*stride_row;
    y += int64_t(sample)*stride_sample_y + channel*stride_channel_y;

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};

    const float2 * x2 = (const float2 *) x;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float2 tmpy = y2[j*stride_col_y2 + col2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid != 0) {
        return;
    }

#pragma unroll
    for (int t = 0; t < n_tokens; ++t) {
        float acc = relu_sum_heads_relu(sumf[t*n_head]);
#pragma unroll
        for (int h = 1; h < n_head; ++h) {
            acc = acc + relu_sum_heads_relu(sumf[t*n_head + h]);
        }
        if (bias) {
            acc = acc + bias[int64_t(sample)*stride_sample_bias + channel*stride_channel_bias + t*stride_col_bias + row];
        }
        dst[int64_t(sample)*stride_sample_dst + channel*stride_channel_dst + t*stride_col_dst + row] = acc;
    }
}

static constexpr int MMVF_RELU_SUM_HEADS_N_HEAD = 4;

bool ggml_cuda_mul_mat_vec_f_relu_sum_heads_supported(const ggml_tensor * mm, const ggml_tensor * rsh) {
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    const ggml_tensor * bias = rsh->src[1];

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
            rsh->type != GGML_TYPE_F32 || (bias && bias->type != GGML_TYPE_F32)) {
        return false;
    }
    if (ggml_get_op_params_i32(rsh, 0) != MMVF_RELU_SUM_HEADS_N_HEAD) {
        return false;
    }
    const int64_t ncols_dst = src1->ne[1];
    if (ncols_dst != MMVF_RELU_SUM_HEADS_N_HEAD && ncols_dst != 2*MMVF_RELU_SUM_HEADS_N_HEAD) {
        return false;
    }
    // no broadcast: every channel and sample of src1 meets its own of src0
    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || rsh->nb[0] != sizeof(float) ||
            (bias && bias->nb[0] != sizeof(float))) {
        return false;
    }
    if (src0->ne[0] % 2 != 0 || (src0->nb[1]/sizeof(float)) % 2 != 0 || (src1->nb[1]/sizeof(float)) % 2 != 0) {
        return false;
    }
    const int64_t block_size = mul_mat_vec_f_block_size(src0->ne[0]);
    if (block_size % 32 != 0 || block_size < 32 || block_size > 256) {
        return false;
    }
    const int64_t ts = sizeof(float);
    return src0->ne[1] <= INT_MAX && src0->ne[2] <= 65535 && src0->ne[3] <= 65535 &&
        src0->nb[1]/ts <= INT_MAX && src1->nb[1]/ts <= INT_MAX && src0->nb[2]/ts <= INT_MAX && src1->nb[2]/ts <= INT_MAX &&
        rsh->nb[1]/ts <= INT_MAX && rsh->nb[2]/ts <= INT_MAX &&
        (!bias || (bias->nb[1]/ts <= INT_MAX && bias->nb[2]/ts <= INT_MAX));
}

template <int ncols_dst, int block_size>
static void mul_mat_vec_f_relu_sum_heads_launch(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * bias,
        ggml_tensor * dst) {
    const int64_t ts = sizeof(float);
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    const dim3 block_nums(src0->ne[1], src0->ne[2], src0->ne[3]);
    const dim3 block_dims(block_size, 1, 1);
    const int nbytes_shared = warp_size*sizeof(float);

    mul_mat_vec_f_relu_sum_heads<ncols_dst, block_size, MMVF_RELU_SUM_HEADS_N_HEAD><<<block_nums, block_dims, nbytes_shared, ctx.stream()>>>(
        (const float *) src0->data, (const float *) src1->data, bias ? (const float *) bias->data : nullptr, (float *) dst->data,
        src0->ne[0]/2, src0->nb[1]/ts, (src1->nb[1]/ts)/2,
        src0->nb[2]/ts, src1->nb[2]/ts, src0->nb[3]/ts, src1->nb[3]/ts,
        dst->nb[1]/ts, dst->nb[2]/ts, dst->nb[3]/ts,
        bias ? bias->nb[1]/ts : 0, bias ? bias->nb[2]/ts : 0, bias ? bias->nb[3]/ts : 0);
    CUDA_CHECK(cudaGetLastError());
}

template <int ncols_dst>
static void mul_mat_vec_f_relu_sum_heads_switch_block_size(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * bias,
        ggml_tensor * dst) {
    switch (mul_mat_vec_f_block_size(src0->ne[0])) {
        case  32: mul_mat_vec_f_relu_sum_heads_launch<ncols_dst,  32>(ctx, src0, src1, bias, dst); break;
        case  64: mul_mat_vec_f_relu_sum_heads_launch<ncols_dst,  64>(ctx, src0, src1, bias, dst); break;
        case  96: mul_mat_vec_f_relu_sum_heads_launch<ncols_dst,  96>(ctx, src0, src1, bias, dst); break;
        case 128: mul_mat_vec_f_relu_sum_heads_launch<ncols_dst, 128>(ctx, src0, src1, bias, dst); break;
        case 160: mul_mat_vec_f_relu_sum_heads_launch<ncols_dst, 160>(ctx, src0, src1, bias, dst); break;
        case 192: mul_mat_vec_f_relu_sum_heads_launch<ncols_dst, 192>(ctx, src0, src1, bias, dst); break;
        case 224: mul_mat_vec_f_relu_sum_heads_launch<ncols_dst, 224>(ctx, src0, src1, bias, dst); break;
        case 256: mul_mat_vec_f_relu_sum_heads_launch<ncols_dst, 256>(ctx, src0, src1, bias, dst); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_f_relu_sum_heads(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * rsh) {
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    const ggml_tensor * bias = rsh->src[1];

    ggml_cuda_assert_src0_is_device_readable(src0);
    GGML_ASSERT(ggml_cuda_mul_mat_vec_f_relu_sum_heads_supported(mm, rsh));


    switch (src1->ne[1]) {
        case MMVF_RELU_SUM_HEADS_N_HEAD:
            mul_mat_vec_f_relu_sum_heads_switch_block_size<MMVF_RELU_SUM_HEADS_N_HEAD>(ctx, src0, src1, bias, rsh);
            break;
        case 2*MMVF_RELU_SUM_HEADS_N_HEAD:
            mul_mat_vec_f_relu_sum_heads_switch_block_size<2*MMVF_RELU_SUM_HEADS_N_HEAD>(ctx, src0, src1, bias, rsh);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
