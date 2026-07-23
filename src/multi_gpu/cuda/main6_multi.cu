#include <cuda_runtime.h>   // for cudaMalloc, cudaFree, etc.
#include <curand_kernel.h>  // for CURAND device RNG
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <x86intrin.h>  // for __rdtsc()

#include <cmath>
#include <cstdint>      // for fixed-width integer types
#include <cub/cub.cuh>  // for CUB algorithms
#include <fstream>      // to read/write files
#include <sstream>      // to use stringstream
#include <stdexcept>

namespace kernel6 {

// Globals
thread_local uint64_t N;
__constant__ uint64_t d_N;
thread_local cudaMemPool_t g_pool;
thread_local cudaStream_t g_stream;
thread_local int global_gpu_id;

bool debug = false;

#define cudaMalloc_log(ptr, size, ...) do_cudaMalloc_log(ptr, size, #ptr, ##__VA_ARGS__)
#define cudaFree_log(ptr, ...) do_cudaFree_log(ptr, #ptr, ##__VA_ARGS__)

void print_GPU_info(bool show_bytes = false) {
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, global_gpu_id);

  size_t reserved_mem = 0;
  size_t used_mem = 0;

  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_mem);
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used_mem);

  if (show_bytes) {
    std::cout << "POOL USAGE: " << used_mem << " bytes / " << reserved_mem << " bytes ";
  } else {
    std::cout << "POOL USAGE: " << used_mem / (1024.0 * 1024.0 * 1024.0) << " GB / "
              << reserved_mem / (1024.0 * 1024.0 * 1024.0) << " GB ";
  }
}

inline std::string strip_ampersand(const std::string& s) {
  if (!s.empty() && s[0] == '&') return s.substr(1);
  return s;
}

template <typename T, typename size_Type>
cudaError_t do_cudaMalloc_log(T** devPtr, size_Type size, std::string label = "", bool verbose = false) {
  cudaError_t status = cudaMallocFromPoolAsync(devPtr, size, g_pool, g_stream);
  if (status != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(status) << std::endl;
    exit(1);
  }
  if (verbose) {
    printf("[cudaMalloc] ");
    print_GPU_info();
    label = strip_ampersand(label);
    std::cout << "\033[31m" << label << " " << (void*)*devPtr << "\033[0m "
              << "Allocating " << size << " bytes, \033[31m" << size / (1024.0 * 1024.0 * 1024.0)
              << "\033[0m GB on GPU\n";
  }
  return status;
}

template <typename T>
cudaError_t do_cudaFree_log(T* devPtr, std::string label = "", bool verbose = false) {
  cudaError_t status = cudaFreeAsync(devPtr, g_stream);
  if (status != cudaSuccess) {
    std::cerr << "cudaFree failed: " << cudaGetErrorString(status) << std::endl;
    exit(1);
  }
  if (verbose) {
    printf("[cudaFree] ");
    print_GPU_info();
    std::cout << "\033[31m" << label << " " << (void*)devPtr << "\033[0m "
              << "Freeing GPU memory\n";
  }
  return status;
}

template <typename A, typename B>
int blocks(A size, B threads) {
  return (int)((size + threads - 1) / threads);
}

void print_binary(uint32_t n) {
  uint32_t maxpow = (uint32_t)1ull << (sizeof(uint32_t) * 8 - 1);
  for (int i = 0; i < int(sizeof(n) * 8); ++i) {
    printf("%u", N & maxpow ? 1 : 0);
    n <<= 1;
  }
}

template <typename T>
__device__ void print_binary_device(T n) {
  T maxpow = (T)1ull << (sizeof(T) * 8 - 1);
  for (int i = 0; i < sizeof(n) * 8; ++i) {
    printf("%u", N & maxpow ? 1 : 0);
    n <<= 1;
  }
}

void print_last_cuda_error(const char* file, int line, const char* func) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cuda error at %s:%d (%s): %s (code %d)\n", file, line, func, cudaGetErrorString(err),
           static_cast<int>(err));
  } else {
    // printf("no cuda error detected at %s:%d (%s)\n", file, line, func);
  }
  cudaDeviceSynchronize();
}

template <typename in_T, typename out_T, typename size_T>
void sum_vector(in_T vector, out_T output, size_t size) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vector, output, size);
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vector, output, size);
  cudaFree_log(d_temp_storage, false);
}

template <typename in_T, typename out_T, typename size_T>
void exclusive_sum(in_T vector, out_T output, size_T size) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, vector, output, size);
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, vector, output, size);
  cudaFree_log(d_temp_storage, false);
}

template <typename in_T, typename out_T, typename size_T>
void inclusive_sum(in_T vector, out_T output, size_T size) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, vector, output, size);
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, vector, output, size);
  cudaFree_log(d_temp_storage, false);
}

template <typename T1, typename T2>
__global__ void check_overflow(const T1* vector, const T2 size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) {
    bool gt = vector[idx + 1] >= vector[idx];
    if (!gt) printf("POTENTIAL OVERFLOW DETECTED\n");
    assert(gt);
  }
}

template <typename T>
__global__ void finalize_sum(T* output, T* last, size_t size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) output[size] = output[size - 1] + last[0];
}

// See for which cases you REALLY need a complete sum
// Output will be "0,inclusive_sum", so output should have size+1
// Safe for in-place operations
template <typename in_T, typename out_T>
// vector can be an ITERATOR or a pointer. output will always be a pointer
void complete_sum(in_T vector, out_T* output, size_t size) {
  out_T* d_last;
  cudaMalloc_log(&d_last, sizeof(out_T));
  thrust::copy_n(thrust::device, vector + size - 1, 1, d_last);  // Works for pointer and also iterator
  exclusive_sum(vector, output, size);
  finalize_sum<<<1, 1>>>(output, d_last, size);
  cudaFree_log(d_last);
  if (debug) {
    check_overflow<<<blocks(size, 256), 256>>>(output, size);
  }
}

template <typename in_T, typename out_T, typename mask_T>
size_t device_select_flagged(in_T d_in, out_T d_out, mask_T d_mask, size_t size) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  size_t* d_num_selected = nullptr;
  cudaMalloc_log(&d_num_selected, sizeof(size_t), false);

  cudaError_t cuda_status =
      cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_mask, d_out, d_num_selected, size);
  if (cuda_status != cudaSuccess) {
    std::cerr << "error: " << cudaGetErrorString(cuda_status) << std::endl;
    exit(1);
  }
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
  cuda_status =
      cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_mask, d_out, d_num_selected, size);
  if (cuda_status != cudaSuccess) {
    std::cerr << "error: " << cudaGetErrorString(cuda_status) << std::endl;
    exit(1);
  }
  size_t h_count;
  cudaMemcpy(&h_count, d_num_selected, sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaFree_log(d_temp_storage);
  cudaFree_log(d_num_selected);
  return h_count;
}

template <typename T>
struct NullableType {
  T value = T(0);
  bool is_null = false;
};

template <typename A, typename B, typename C>
__device__ NullableType<A> binary_search(A idx_start,
                                         A idx_end,  // exclusive
                                         B* values, C target) {
  NullableType<A> result;
  while (idx_start < idx_end) {
    A middle_point = idx_start + (idx_end - idx_start) / 2;
    if (values[middle_point] < target)
      idx_start = middle_point + 1;
    else if (target < values[middle_point])
      idx_end = middle_point;
    else {
      result.value = middle_point;
      return result;
    }
  }
  result.is_null = true;
  return result;
}

template <typename A, typename B, typename C>
__device__ A binary_search_bounds(A idx_start,
                                  A idx_end,  // exclusive (just pass len(values) in many cases)
                                  B* values, C target) {
  while (idx_start < idx_end) {
    A middle_point = idx_start + (idx_end - idx_start) / 2;
    if (values[middle_point] <= target)
      idx_start = middle_point + 1;
    else
      idx_end = middle_point;
  }
  return idx_start - 1;
}

__global__ void find_global_idxs(const uint64_t* d_cols_rows_zipped, uint64_t* global_idxs, const uint32_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) {
    uint32_t c = (uint32_t)(d_cols_rows_zipped[idx] >> 32);
    uint32_t r = (uint32_t)d_cols_rows_zipped[idx];
    global_idxs[idx] = r + (c / 32) * d_N;
  }
}

__global__ void find_global_idxs_u(const uint64_t* d_cols_rows_zipped, uint64_t* global_idxs, const uint32_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) {
    uint32_t c = (uint32_t)(d_cols_rows_zipped[idx] >> 32);
    uint32_t r = (uint32_t)d_cols_rows_zipped[idx];
    // printf("%llu %llu\n", idx, (uint64_t)size);
    global_idxs[idx] = c * (d_N / 32) + r / 32;
  }
}

template <typename T, typename T_size>
__host__ uint32_t unique(T*& d_in, T_size num_items) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  T* d_out;
  cudaMalloc_log(&d_out, num_items * sizeof(T));
  uint32_t* d_num_selected;
  cudaMalloc_log(&d_num_selected, sizeof(uint32_t));

  cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items);
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected, num_items);
  cudaFree_log(d_temp_storage);
  cudaFree_log(d_in);
  d_in = d_out;

  uint32_t h_count;
  cudaMemcpy(&h_count, d_num_selected, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree_log(d_num_selected);
  return h_count;
}

template <typename T>
__host__ void radix_sort(T*& input, const size_t size) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  T* output;
  cudaMalloc_log(&output, size * sizeof(T));

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, input, output, size);
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, input, output, size);

  cudaFree_log(input);
  cudaFree_log(d_temp_storage);

  input = output;
}

template <typename keyt, typename valuet>
__host__ void radix_sort_pairs(keyt*& global_idxs, valuet*& d_cols, const uint32_t size) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  keyt* global_idxs_out;
  valuet* d_cols_out;
  cudaMalloc_log(&global_idxs_out, size * sizeof(keyt));
  cudaMalloc_log(&d_cols_out, size * sizeof(valuet));

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, global_idxs, global_idxs_out, d_cols, d_cols_out,
                                  size);
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, global_idxs, global_idxs_out, d_cols, d_cols_out,
                                  size);

  cudaFree_log(global_idxs);
  cudaFree_log(d_cols);
  cudaFree_log(d_temp_storage, false);

  global_idxs = global_idxs_out;
  d_cols = d_cols_out;
}

struct Or_Custom {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a | b;
  }
};

struct Popc_Custom {
  template <typename T>
  __device__ T operator()(const T& a) const {
    return __popc(a);
  }
};

template <typename inputiteratort, typename outputiteratort, typename offsetiteratort>
__host__ void device_segmented_reduce_sum(inputiteratort d_input, outputiteratort d_output, uint32_t num_segments,
                                          offsetiteratort d_offsets_it) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, num_segments, d_offsets_it,
                                  d_offsets_it + 1);
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, num_segments, d_offsets_it,
                                  d_offsets_it + 1);
  cudaFree_log(d_temp_storage);
}

template <typename keysinputiteratort, typename valuesinputiteratort, typename numitemst, typename reductionopt>
__host__ std::tuple<uint64_t*, uint32_t*, uint32_t> reduce_by_key(keysinputiteratort d_keys_in,
                                                                  valuesinputiteratort d_values_in, numitemst size,
                                                                  reductionopt reduction_op) {
  uint64_t* d_keys_out;
  uint32_t *d_values_out, *d_num_runs_out;
  cudaMalloc_log(&d_keys_out, size * sizeof(uint64_t));
  cudaMalloc_log(&d_values_out, size * sizeof(uint32_t));
  cudaMalloc_log(&d_num_runs_out, sizeof(uint32_t));
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                                 d_num_runs_out, reduction_op, size);
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                                 d_num_runs_out, reduction_op, size);
  cudaFree_log(d_temp_storage);
  // send d_num_runs out to host
  uint32_t num_runs_out;
  cudaMemcpy(&num_runs_out, d_num_runs_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree_log(d_num_runs_out);
  return {d_keys_out, d_values_out, num_runs_out};
}

template <typename ReductionOpt>
__host__ uint32_t compress_mask(uint64_t*& global_idxs, uint32_t*& d_mask_l, const uint32_t size,
                                ReductionOpt reduction_op) {
  auto [global_idxs_out, d_mask_lout, num_runs_out] = reduce_by_key(global_idxs, d_mask_l, size, reduction_op);

  cudaFree_log(global_idxs);
  cudaFree_log(d_mask_l);
  d_mask_l = d_mask_lout;
  global_idxs = global_idxs_out;

  return num_runs_out;
}

template <typename T>
__global__ void compute_histogram(const T* d_input, uint32_t* d_hist, size_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) atomicAdd(&d_hist[d_input[idx]], 1);
}

// alternative for my histogram:
template <typename T>
__global__ void compute_histogram_l(const T* d_input, uint32_t* d_hist, size_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) atomicAdd(&d_hist[d_input[idx] / d_N], 1);
}

template <typename T>
__global__ void compute_histogram_u(const T* d_input, uint32_t* d_hist, size_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) {
    atomicAdd(&d_hist[d_input[idx] / (d_N / 32)], 1);
  }
}

template <typename T>
__global__ void compute_histogram_popc(const T* d_input, uint32_t* d_hist, size_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) atomicAdd(&d_hist[uint32_t(d_input[idx] >> 32)], 1);
}

struct divide_by_N {
  template <typename T>
  __device__ T operator()(const T& x) const {
    return x / T(d_N);
  }
};

struct divide_by_N_32 {
  template <typename T>
  __device__ T operator()(const T& x) const {
    return x / T(d_N / 32);
  }
};

template <typename inT, typename outT>
__global__ void modulo_N(const inT* in, outT* out, const uint32_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) out[idx] = in[idx] % d_N;
}

template <typename inT, typename outT>
__global__ void modulo_N_32(const inT* in, outT* out, const uint32_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) out[idx] = in[idx] % (d_N / 32);
}

__global__ void get_rows(const uint64_t* d_cols_rows_zipped, uint32_t* d_out, size_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) d_out[idx] = uint32_t(d_cols_rows_zipped[idx]);
}

__inline__ __device__ uint32_t warpSum32(uint32_t val) {
  // full warp reduction
  for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down_sync(0xffffffff, val, offset);
  return val;  // thread 0 has total sum
}

template <typename T>
__global__ void compute_binary_popcount_mma_9(const uint32_t* d_mask_u, T* d_col_id_u, T* d_set_ptr_u,
                                              const uint32_t* d_mask_l, T* d_col_id_l, T* d_set_ptr_l, T* d_idxs,
                                              uint64_t* d_result, const uint32_t* popc_array, const uint64_t offset,
                                              uint64_t total_num_warps, uint64_t* d_warps_per_i) {
  uint64_t real_idx = 4 * (uint64_t)blockIdx.x + threadIdx.y + offset;
  uint64_t idx;
  if (real_idx >= total_num_warps)
    idx = total_num_warps - 1;
  else
    idx = real_idx;

  uint32_t i = binary_search_bounds(0u, (uint32_t)d_N, d_warps_per_i, idx);
  assert(i < d_N);
  uint64_t block_dim_x = (popc_array[i + 1] - popc_array[i] + 31) / 32;
  assert(block_dim_x > 0ULL);
  assert(idx >= d_warps_per_i[i]);
  uint64_t block_x = (idx - d_warps_per_i[i]) % block_dim_x;
  uint64_t block_y = (idx - d_warps_per_i[i]) / block_dim_x;

  __align__(16) __shared__ uint32_t dA[32];
  __align__(16) __shared__ uint32_t dB[4 * 32];
  __align__(16) __shared__ int32_t dD[4 * 64];

  // Fill dA
  size_t col = d_set_ptr_u[i] + block_y;
  if (threadIdx.x < 8 && threadIdx.y < 4) {
    if (threadIdx.x == 0 && real_idx == idx) {
      dA[4 * threadIdx.x + threadIdx.y] = d_mask_u[col];
    } else
      dA[4 * threadIdx.x + threadIdx.y] = 0;
  }

  // Fill dB
  if (threadIdx.x < 32 && threadIdx.y < 4) {
    size_t idxx = popc_array[i] + 32 * block_x + threadIdx.x;
    size_t unrolled_id = 32 * (threadIdx.x / 8) + 4 * (threadIdx.x % 8) + threadIdx.y;
    if (idxx < popc_array[i + 1] && real_idx == idx) {
      uint32_t id = d_idxs[idxx];
      uint32_t c = d_col_id_u[col];
      NullableType<T> result = binary_search(d_set_ptr_l[c], d_set_ptr_l[c + 1], d_col_id_l, id);
      dB[unrolled_id] = !result.is_null ? d_mask_l[result.value] : 0;
    } else
      dB[unrolled_id] = 0;
  }
  __syncthreads();

  asm volatile(
      R"(
                         // Execute tensor core
                         .reg .b32 a, b, c<2>, d<2>, e;
                         mov.b32 c0, 0;
                         mov.b32 c1, 0;
                         wmma.load.a.sync.aligned.m8n8k128.shared.row.b1 {a}, [%0];
                         wmma.load.b.sync.aligned.m8n8k128.shared.col.b1 {b}, [%1];
                         wmma.mma.and.popc.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32 {d0,d1}, {a}, {b}, {c0,c1};

                         // Store to dD
                         wmma.store.d.sync.aligned.m8n8k128.shared.row.s32 [%2], {d0,d1};
        )"
      :
      : "l"(dA), "l"(dB + 32 * threadIdx.y), "l"(dD + 64 * threadIdx.y)
      : "memory");
  __syncthreads();

  if (threadIdx.y == 0) {
    uint32_t val = dD[threadIdx.x % 8 + (threadIdx.x / 8) * 64];
    val = warpSum32(val);
    if (threadIdx.x == 0) {
      atomicAdd((unsigned long long int*)d_result, (unsigned long long int)val);
    }
  }
}

__global__ void create_non_compact_mask_l(const uint64_t* d_cols_rows_zipped, uint32_t* d_mask_l, const uint32_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) {
    uint32_t c = uint32_t(d_cols_rows_zipped[idx] >> 32);
    d_mask_l[idx] = uint32_t(1) << (31 - c % 32);
  }
}

__global__ void create_non_compact_mask_u(const uint64_t* d_cols_rows_zipped, uint32_t* d_mask_u, const uint32_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) {
    uint32_t r = uint32_t(d_cols_rows_zipped[idx]);
    d_mask_u[idx] = uint32_t(1) << (31 - r % 32);
  }
}

template <typename T>
__host__ uint32_t run_length_encode(T d_input, uint32_t*& d_counts_out, const uint32_t size) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  uint32_t* d_unique_temp;
  uint32_t* d_num_runs_temp;
  cudaMalloc_log(&d_unique_temp, size * sizeof(uint32_t));
  cudaMalloc_log(&d_num_runs_temp, sizeof(uint32_t));
  cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_input, d_unique_temp, d_counts_out,
                                     d_num_runs_temp, size);
  cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_input, d_unique_temp, d_counts_out,
                                     d_num_runs_temp, size);
  uint32_t num_runs;
  cudaMemcpy(&num_runs, d_num_runs_temp, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree_log(d_temp_storage);
  cudaFree_log(d_unique_temp);
  cudaFree_log(d_num_runs_temp);
  return num_runs;
}

__global__ void compute_n_warps_per_i(uint64_t* d_warps_per_i, uint32_t* d_set_ptr_u, uint32_t* popc_array,
                                      uint32_t size) {
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  if (idx < size) {
    d_warps_per_i[idx] = (uint64_t)(d_set_ptr_u[idx + 1] - d_set_ptr_u[idx]) *
                         (uint64_t)((popc_array[idx + 1] - popc_array[idx] + 31) / 32);
  }
}

extern "C" void init_gpu_kernel6(int gpu_id) {
  global_gpu_id = gpu_id;
  cudaSetDevice(global_gpu_id);
  cudaDeviceGetDefaultMemPool(&g_pool, global_gpu_id);
  g_stream = 0;
  // Memory in pool forever
  uint64_t threshold = UINT64_MAX;
  cudaMemPoolSetAttribute(g_pool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

extern "C" uint64_t count_partial_triangles_off_diagonal(uint64_t** cols_rows_zipped_COLS, uint64_t* nnz_COLS_ptr,
                                                         uint32_t num_ptrs_cols, uint64_t** cols_rows_zipped_IDXS,
                                                         uint64_t* nnz_IDXS_ptr, uint32_t num_ptrs_idxs,
                                                         uint64_t num_nodes) {
  uint64_t nnz_COLS = nnz_COLS_ptr[num_ptrs_cols];  // Safe since nnz_COLS_ptr is cumulative and has one extra value
  uint64_t* d_cols_rows_zipped_COLS;
  cudaMalloc_log(&d_cols_rows_zipped_COLS, nnz_COLS * sizeof(uint64_t));
  for (int i = 0; i < num_ptrs_cols; ++i) {
    uint64_t offset = nnz_COLS_ptr[i];
    uint64_t nnz_block = nnz_COLS_ptr[i + 1] - nnz_COLS_ptr[i];
    cudaMemcpy(d_cols_rows_zipped_COLS + offset, cols_rows_zipped_COLS[i], nnz_block * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
  }

  uint64_t nnz_IDXS = nnz_IDXS_ptr[num_ptrs_idxs];  // Safe since nnz_IDXS_ptr is cumulative and has one extra value
  uint64_t* d_cols_rows_zipped_IDXS;
  cudaMalloc_log(&d_cols_rows_zipped_IDXS, nnz_IDXS * sizeof(uint64_t));
  for (int i = 0; i < num_ptrs_idxs; ++i) {
    uint64_t offset = nnz_IDXS_ptr[i];
    uint64_t nnz_block = nnz_IDXS_ptr[i + 1] - nnz_IDXS_ptr[i];
    cudaMemcpy(d_cols_rows_zipped_IDXS + offset, cols_rows_zipped_IDXS[i], nnz_block * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
  }

  N = 32 * ((num_nodes + 31) / 32);
  cudaMemcpyToSymbol(d_N, &N, sizeof(uint64_t));

  radix_sort(d_cols_rows_zipped_COLS, nnz_COLS);
  nnz_COLS = unique(d_cols_rows_zipped_COLS, nnz_COLS);

  radix_sort(d_cols_rows_zipped_IDXS, nnz_IDXS);
  // Here we cut the number of ones that the matrix processes.
  // This turns out to be very useful, because d_idxs do not need ALL the ones, only those reaching the lowest block of
  // cols
  nnz_IDXS = unique(d_cols_rows_zipped_IDXS, nnz_IDXS);

  // -------- MATRIX L -------- //
  uint64_t* global_idxs;
  cudaMalloc_log(&global_idxs, nnz_IDXS * sizeof(uint64_t));  // Global idxs specific for IDXS
  uint32_t* d_mask_l;
  cudaMalloc_log(&d_mask_l, nnz_IDXS * sizeof(uint32_t));
  create_non_compact_mask_l<<<blocks(nnz_IDXS, 256), 256>>>(d_cols_rows_zipped_IDXS, d_mask_l, nnz_IDXS);
  // create global_idxs
  find_global_idxs<<<blocks(nnz_IDXS, 256), 256>>>(d_cols_rows_zipped_IDXS, global_idxs, nnz_IDXS);
  // sort d_mask_l using global_idxs
  radix_sort_pairs(global_idxs, d_mask_l, nnz_IDXS);
  // compress elements of mask_d_l_ with same global_idx using or operation
  uint32_t compressed_size_l = 0;
  compressed_size_l = compress_mask(global_idxs, d_mask_l, nnz_IDXS, Or_Custom{});
  // create d_row_id_l
  uint32_t* d_row_id_l;
  cudaMalloc_log(&d_row_id_l, compressed_size_l * sizeof(uint32_t));
  modulo_N<<<blocks(compressed_size_l, 256), 256>>>(global_idxs, d_row_id_l, compressed_size_l);  // compressed rows
  // create d_set_ptr_l
  uint32_t* d_set_ptr_l;
  cudaMalloc_log(&d_set_ptr_l, (N / 32 + 1) * sizeof(uint32_t));
  cudaMemset(d_set_ptr_l, 0, (N / 32 + 1) * sizeof(uint32_t));
  compute_histogram_l<<<blocks(compressed_size_l, 256), 256>>>(
      // thrust::make_transform_iterator(global_idxs, divide_by_n()), // compressed columns (no need to store them)
      global_idxs, d_set_ptr_l, compressed_size_l);
  cudaFree_log(global_idxs);
  /*inclusive_sum(d_set_ptr_l, d_set_ptr_l+1, N/32);
  cudaMemset(d_set_ptr_l, 0, sizeof(uint32_t));*/
  complete_sum(d_set_ptr_l, d_set_ptr_l, N / 32);
  // -------- MATRIX L -------- //

  // -------- MATRIX U -------- //
  // Create d_mask_u
  uint32_t* d_mask_u;
  cudaMalloc_log(&d_mask_u, nnz_COLS * sizeof(uint32_t));
  create_non_compact_mask_u<<<blocks(nnz_COLS, 256), 256>>>(d_cols_rows_zipped_COLS, d_mask_u, nnz_COLS);
  // Create d_idxs
  uint32_t* d_idxs;
  cudaMalloc_log(&d_idxs, nnz_IDXS * sizeof(uint32_t));
  get_rows<<<blocks(nnz_IDXS, 256), 256>>>(d_cols_rows_zipped_IDXS, d_idxs, nnz_IDXS);
  // Create global_idxs_u
  cudaMalloc_log(&global_idxs, nnz_COLS * sizeof(uint64_t));  // Global idxs specific for COLS
  find_global_idxs_u<<<blocks(nnz_COLS, 256), 256>>>(d_cols_rows_zipped_COLS, global_idxs, nnz_COLS);
  cudaFree_log(d_cols_rows_zipped_COLS);
  // Sort d_mask_u using global_idxs
  radix_sort_pairs(global_idxs, d_mask_u, nnz_COLS);
  // Compress elements of mask_d_u_ with SAME global_idx using OR operation
  uint32_t compressed_size_u = 0;
  compressed_size_u = compress_mask(global_idxs, d_mask_u, nnz_COLS, Or_Custom{});
  // Create d_col_id_u
  uint32_t* d_col_id_u;
  cudaMalloc_log(&d_col_id_u, compressed_size_u * sizeof(uint32_t));
  modulo_N_32<<<blocks(compressed_size_u, 256), 256>>>(global_idxs, d_col_id_u, compressed_size_u);  // Compressed cols
  // Create d_set_ptr_u
  uint32_t* d_set_ptr_u;
  cudaMalloc_log(&d_set_ptr_u, (N + 1) * sizeof(uint32_t));
  /*uint32_t non_zero_rows = run_length_encode( // We use RLE instead of histogram to ignore ZEROs in rows (which are
  completely unnecessary) thrust::make_transform_iterator(global_idxs, divide_by_N_32()), // Compressed rows (no need to
  store them) d_set_ptr_u, compressed_size_u
  );*/
  cudaMemset(d_set_ptr_u, 0, N * sizeof(uint32_t));
  compute_histogram_u<<<blocks(compressed_size_u, 256), 256>>>(
      // thrust::make_transform_iterator(global_idxs, divide_by_N_32()), // Compressed rows (no need to store them)
      global_idxs, d_set_ptr_u, compressed_size_u);
  cudaFree_log(global_idxs);
  complete_sum(d_set_ptr_u, d_set_ptr_u, N);
  // Create popc_array
  uint32_t* popc_array;
  cudaMalloc_log(&popc_array, (N + 1) * sizeof(uint32_t));
  cudaMemset(popc_array, 0, N * sizeof(uint32_t));
  compute_histogram_popc<<<blocks(nnz_IDXS, 256), 256>>>(d_cols_rows_zipped_IDXS, popc_array, nnz_IDXS);
  cudaFree_log(d_cols_rows_zipped_IDXS);
  complete_sum(popc_array, popc_array, N);
  // -------- MATRIX U -------- //

  uint64_t* d_warps_per_i;
  cudaMalloc_log(&d_warps_per_i, (N + 1) * sizeof(uint64_t));
  compute_n_warps_per_i<<<blocks(N, 256), 256>>>(d_warps_per_i, d_set_ptr_u, popc_array, N);
  complete_sum(d_warps_per_i, d_warps_per_i, N);
  uint64_t total_num_warps;
  cudaMemcpy(&total_num_warps, d_warps_per_i + N, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  uint64_t total_num_blocks = (total_num_warps + 3ULL) / 4ULL;  // A block will be 4 warps

  uint64_t triangle_count = 0, *d_result;
  cudaMalloc_log(&d_result, sizeof(uint64_t));
  cudaMemset(d_result, 0, sizeof(uint64_t));

  uint64_t offset = 0ULL;
  int max_num_blocks_allowed = INT_MAX;
  uint32_t number_of_launches =
      (uint32_t)((total_num_blocks + (uint64_t)max_num_blocks_allowed - 1) / (uint64_t)max_num_blocks_allowed);
  // if (number_of_launches != 1u) printf("Launching %u kernels...\n", number_of_launches);

  for (uint32_t launch = 0; launch < number_of_launches; ++launch) {
    int blocksPerGrid =
        (launch != number_of_launches - 1) ? max_num_blocks_allowed : total_num_blocks % max_num_blocks_allowed;
    dim3 threadsPerBlock(32, 4);
    compute_binary_popcount_mma_9<<<blocksPerGrid, threadsPerBlock>>>(
        d_mask_u, d_col_id_u, d_set_ptr_u, d_mask_l, d_row_id_l, d_set_ptr_l, d_idxs, d_result, popc_array, offset,
        total_num_warps, d_warps_per_i);
    offset += 4 * (uint64_t)blocksPerGrid;
  }

  cudaFree_log(d_mask_u);
  cudaFree_log(d_col_id_u);
  cudaFree_log(d_set_ptr_u);
  cudaFree_log(popc_array);
  cudaFree_log(d_idxs);
  cudaFree_log(d_mask_l);
  cudaFree_log(d_row_id_l);
  cudaFree_log(d_set_ptr_l);
  cudaFree_log(d_warps_per_i);

  cudaMemcpy(&triangle_count, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaFree_log(d_result);

  print_last_cuda_error(__FILE__, __LINE__, __func__);

  cudaStreamSynchronize(g_stream);

  return triangle_count;
}

}  // namespace kernel6
