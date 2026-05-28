/**
 * Stable variant of the DeepSeek NSA top-k selector.
 *
 * The original topk.cu is optimized for throughput and uses atomic append order
 * for both selected entries and final tie buckets. This file keeps a separate
 * implementation with an explicit total order:
 *
 *   larger score wins; exact score ties are broken by smaller local index.
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr int kRadix = 256;
constexpr int kKeyBytes = 8;

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

__device__ __forceinline__ uint32_t float_to_ordered_uint(float x) {
  // Keep +0.0 and -0.0 in the same tie bucket so index order decides ties.
  if (x == 0.0f) x = 0.0f;
  // Scores are expected to be finite. If a NaN sneaks in, keep it at the bottom
  // instead of letting its payload dominate the ordering.
  if (x != x) return 0;

  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint64_t stable_topk_key(float score, uint32_t local_idx) {
  const uint64_t score_key = static_cast<uint64_t>(float_to_ordered_uint(score));
  const uint64_t idx_key = static_cast<uint64_t>(0xFFFFFFFFu - local_idx);
  return (score_key << 32) | idx_key;
}

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void
naive_topk_transform(int32_t length, int32_t* __restrict__ dst_page_table, const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

template <int N>
__device__ void bitonic_sort_pairs_ascending(uint64_t* keys, int32_t* indices) {
  const auto tid = threadIdx.x;
  static_assert((N & (N - 1)) == 0, "bitonic sort requires power-of-two N");

  for (uint32_t k = 2; k <= N; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      for (uint32_t i = tid; i < N; i += kThreadsPerBlock) {
        const uint32_t ixj = i ^ j;
        if (ixj > i) {
          const bool ascending = (i & k) == 0;
          const auto key_i = keys[i];
          const auto key_j = keys[ixj];
          const bool should_swap = ascending ? (key_i > key_j) : (key_i < key_j);
          if (should_swap) {
            keys[i] = key_j;
            keys[ixj] = key_i;
            const auto idx_i = indices[i];
            indices[i] = indices[ixj];
            indices[ixj] = idx_i;
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ void
stable_topk_cuda_tl(const float* __restrict__ input, int32_t* __restrict__ index, int32_t row_start, int32_t length) {
  const auto tx = threadIdx.x;

  alignas(128) __shared__ uint32_t s_hist[kRadix];
  alignas(128) __shared__ uint64_t s_prefix;
  alignas(128) __shared__ uint32_t s_remaining;
  alignas(128) __shared__ uint32_t s_selected_count;
  alignas(128) __shared__ uint64_t s_selected_keys[TopK];
  alignas(128) __shared__ int32_t s_selected_indices[TopK];

  uint64_t prefix = 0;
  uint32_t prefix_bits = 0;
  uint32_t remaining = TopK;

#pragma unroll
  for (int pass = 0; pass < kKeyBytes; ++pass) {
    if (tx < kRadix) {
      s_hist[tx] = 0;
    }
    __syncthreads();

    const int shift = 56 - pass * 8;
    for (int idx = tx; idx < length; idx += kThreadsPerBlock) {
      const auto key = stable_topk_key(input[row_start + idx], static_cast<uint32_t>(idx));
      if (prefix_bits == 0 || (key >> (64 - prefix_bits)) == prefix) {
        const auto bin = static_cast<uint32_t>((key >> shift) & 0xFFu);
        atomicAdd(&s_hist[bin], 1);
      }
    }
    __syncthreads();

    if (tx == 0) {
      uint32_t count_greater = 0;
      uint32_t threshold_bin = 0;
      uint32_t next_remaining = remaining;
      for (int bin = kRadix - 1; bin >= 0; --bin) {
        const auto count = s_hist[bin];
        if (count_greater + count >= remaining) {
          threshold_bin = static_cast<uint32_t>(bin);
          next_remaining = remaining - count_greater;
          break;
        }
        count_greater += count;
      }
      s_prefix = (prefix << 8) | static_cast<uint64_t>(threshold_bin);
      s_remaining = next_remaining;
    }
    __syncthreads();

    prefix = s_prefix;
    remaining = s_remaining;
    prefix_bits += 8;
  }

  const auto threshold_key = prefix;

  for (int i = tx; i < TopK; i += kThreadsPerBlock) {
    s_selected_keys[i] = 0;
    s_selected_indices[i] = -1;
  }
  if (tx == 0) {
    s_selected_count = 0;
  }
  __syncthreads();

  for (int idx = tx; idx < length; idx += kThreadsPerBlock) {
    const auto key = stable_topk_key(input[row_start + idx], static_cast<uint32_t>(idx));
    if (key >= threshold_key) {
      const auto pos = atomicAdd(&s_selected_count, 1);
      if (C10_LIKELY(pos < TopK)) {
        s_selected_keys[pos] = key;
        s_selected_indices[pos] = idx;
      }
    }
  }
  __syncthreads();

  bitonic_sort_pairs_ascending<TopK>(s_selected_keys, s_selected_indices);

  for (int i = tx; i < TopK; i += kThreadsPerBlock) {
    index[i] = s_selected_indices[TopK - 1 - i];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void topk_stable_kernel(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(indice, length);
  } else {
    return stable_topk_cuda_tl(score, indice, row_start, length);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void topk_transform_stable_decode_kernel(
    const FastTopKParams params,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    const int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_transform(length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int32_t s_indices[TopK];
    stable_topk_cuda_tl(score, s_indices, 0, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void topk_transform_stable_prefill_kernel(
    const FastTopKParams params,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    const int64_t src_stride,
    const int32_t* __restrict__ cu_seqlens_q,
    const int64_t prefill_bs) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int32_t s_indices[TopK];
    stable_topk_cuda_tl(score, s_indices, row_start, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void topk_transform_stable_prefill_ragged_kernel(
    const FastTopKParams params,
    int32_t* __restrict__ topk_indices_ragged,
    const int32_t* __restrict__ topk_indices_offset) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto offset = topk_indices_offset[bid];

  if (length <= TopK) {
    return naive_topk_transform_ragged(length, dst_indices_entry, offset);
  } else {
    __shared__ int32_t s_indices[TopK];
    stable_topk_cuda_tl(score, s_indices, row_start, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_indices_entry[idx_0] = pos_0 + offset;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_indices_entry[idx_1] = pos_1 + offset;
  }
}

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt);
void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt);
void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt);

namespace {

bool stable_topk_enabled() {
  const char* value = std::getenv("SGLANG_STABLE_TOPK");
  if (value == nullptr) return false;

  auto lower = [](char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; };
  auto equals = [&](const char* target) {
    for (size_t i = 0;; ++i) {
      if (lower(value[i]) != target[i]) return false;
      if (target[i] == '\0') return true;
    }
  };

  return equals("1") || equals("true") || equals("yes") || equals("y");
}

}  // namespace

void fast_topk_stable_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, row_starts_opt, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  topk_stable_kernel<<<grid, block, 0, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "stable topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_stable_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  if (is_decode) {
    topk_transform_stable_decode_kernel<<<grid, block, 0, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
  } else {
    topk_transform_stable_prefill_kernel<<<grid, block, 0, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "stable topk transform kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_stable_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);
  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  topk_transform_stable_prefill_ragged_kernel<<<grid, block, 0, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "stable ragged topk transform kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_dispatch_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  if (stable_topk_enabled()) {
    fast_topk_stable_interface(score, indices, lengths, row_starts_opt);
    return;
  }
  fast_topk_interface(score, indices, lengths, row_starts_opt);
}

void fast_topk_transform_dispatch_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  if (stable_topk_enabled()) {
    fast_topk_transform_stable_interface(score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts_opt);
    return;
  }
  fast_topk_transform_interface(score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts_opt);
}

void fast_topk_transform_ragged_dispatch_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  if (stable_topk_enabled()) {
    fast_topk_transform_ragged_stable_interface(
        score, lengths, topk_indices_ragged, topk_indices_offset, row_starts_opt);
    return;
  }
  fast_topk_transform_ragged_interface(score, lengths, topk_indices_ragged, topk_indices_offset, row_starts_opt);
}
