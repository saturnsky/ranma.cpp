#include "expert-profiler.cuh"

namespace ggml_cuda_expert {

static __global__ void profile_ids_kernel(
        const int32_t * __restrict__ ids, const int n_rows, const int row_stride, const int n_used, const int n_experts,
        uint32_t * __restrict__ counts, const size_t bank_stride, const uint32_t n_banks,
        const profile_selection * __restrict__ selection) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_rows*n_used) {
        return;
    }
    const profile_selection sel = *selection;
    if (sel.bank >= n_banks) {
        return;
    }
    const int row = i/n_used;
    if (row < sel.row_begin || row >= sel.row_end) {
        return;
    }
    const int k  = i - row*n_used;
    const int id = ids[row*row_stride + k];
    if ((unsigned) id < (unsigned) n_experts) {
        atomicAdd(&counts[size_t(sel.bank)*bank_stride + id], 1U);
    }
}

static __global__ void set_selection_kernel(profile_selection * target, const profile_selection value) {
    *target = value;
}

void launch_profile_ids(
        const int32_t * ids, int n_rows, int row_stride, int n_used, int n_experts,
        uint32_t * counts, size_t bank_stride, uint32_t n_banks, const profile_selection * selection, cudaStream_t stream) {
    const int n = n_rows*n_used;
    if (n <= 0) {
        return;
    }
    const ggml_cuda_kernel_launch_params launch(dim3((n + 127)/128), dim3(128), 0, stream);
    ggml_cuda_kernel_launch(profile_ids_kernel, launch, ids, n_rows, row_stride, n_used, n_experts,
        counts, bank_stride, n_banks, selection);
}

profiler::profiler(const geometry & geo, uint32_t n_banks) :
    n_banks_(n_banks), n_counts_(geo.n_counts()) {
}

profiler::~profiler() {
    if (device_ >= 0) {
        ggml_cuda_set_device(device_);
    }
    if (counts_) {
        (void) cudaFree(counts_);
    }
    if (selection_) {
        (void) cudaFree(selection_);
    }
}

size_t profiler::device_bytes() const {
    return counts_ ? n_counts_*sizeof(uint32_t)*n_banks_ + sizeof(profile_selection) : 0;
}

bool profiler::allocate(int device) {
    if (counts_ != nullptr || n_counts_ == 0 || n_banks_ == 0) {
        return counts_ != nullptr;
    }
    device_ = device;
    ggml_cuda_set_device(device_);
    const size_t bytes = n_counts_*sizeof(uint32_t)*n_banks_;
    if (cudaMalloc((void **) &counts_, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        counts_ = nullptr;
        return false;
    }
    if (cudaMalloc((void **) &selection_, sizeof(profile_selection)) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFree(counts_);
        counts_ = nullptr;
        selection_ = nullptr;
        return false;
    }
    CUDA_CHECK(cudaMemset(counts_, 0, bytes));
    CUDA_CHECK(cudaMemcpy(selection_, &last_selection_, sizeof(profile_selection), cudaMemcpyHostToDevice));
    return true;
}

bool profiler::select(int32_t row_begin, int32_t row_end, uint32_t bank, cudaStream_t stream) {
    if (selection_ == nullptr) {
        return false;
    }
    profile_selection next = { row_begin, row_end, bank, 0 };
    if (row_end <= row_begin || bank >= n_banks_) {
        // canonical "nothing" so that repeated empty selections do not launch
        next = { -1, -1, 0xFFFFFFFFu, 0 };
    }
    if (next.row_begin == last_selection_.row_begin && next.row_end == last_selection_.row_end &&
            next.bank == last_selection_.bank) {
        return true;
    }
    ggml_cuda_set_device(device_);
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(1), 0, stream);
    ggml_cuda_kernel_launch(set_selection_kernel, launch, selection_, next);
    last_selection_ = next;
    return true;
}

bool profiler::read_bank(uint32_t bank, std::vector<uint64_t> & out, bool zero_after) {
    if (counts_ == nullptr || bank >= n_banks_) {
        return false;
    }
    ggml_cuda_set_device(device_);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    std::vector<uint32_t> current(n_counts_);
    uint32_t * base = counts_ + size_t(bank)*n_counts_;
    if (cudaMemcpy(current.data(), base, n_counts_*sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (zero_after) {
        CUDA_CHECK(cudaMemset(base, 0, n_counts_*sizeof(uint32_t)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    out.assign(current.begin(), current.end());
    return true;
}

bool profiler::zero_bank(uint32_t bank) {
    if (counts_ == nullptr || bank >= n_banks_) {
        return false;
    }
    ggml_cuda_set_device(device_);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    CUDA_CHECK(cudaMemset(counts_ + size_t(bank)*n_counts_, 0, n_counts_*sizeof(uint32_t)));
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

} // namespace ggml_cuda_expert
