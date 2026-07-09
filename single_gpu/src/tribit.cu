#include <cuda_runtime.h>       // for cudaMalloc, cudaFree, etc.
#include <curand_kernel.h>      // for CURAND device RNG
#include <cub/cub.cuh>          // for CUB algorithms
#include <cstdint>              // for fixed-width integer types
#include <x86intrin.h>          // for __rdtsc()
#include <fstream>              // to read/write files
#include <sstream>              // to use stringstream
#include <stdexcept>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cmath>
#include <thrust/count.h>
#include <thrust/functional.h>

#include <thrust/version.h>
#include <cub/version.cuh>

uint64_t N;
__constant__ uint64_t d_N;

#define cudaMalloc_log(ptr, size, ...) do_cudaMalloc_log(ptr, size, #ptr, ##__VA_ARGS__)
#define cudaFree_log(ptr, ...) do_cudaFree_log(ptr, #ptr, ##__VA_ARGS__)

void print_GPU_info() {
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);

    size_t reserved_mem = 0;
    size_t used_mem = 0;

    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_mem);
    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used_mem);

    std::cout << "POOL USAGE: "
              << used_mem / (1024.0 * 1024.0 * 1024.0) << " GB / "
              << reserved_mem / (1024.0 * 1024.0 * 1024.0) << " GB ";
}

inline std::string strip_ampersand(const std::string& s) {
    if (!s.empty() && s[0] == '&') return s.substr(1);
    return s;
}

template <typename T, typename size_Type>
cudaError_t do_cudaMalloc_log(T** devPtr, size_Type size, std::string label = "", bool verbose = false) {
        if (verbose) {
                printf("[cudaMalloc] ");
                print_GPU_info();
                label = strip_ampersand(label);
                std::cout << "\033[31m" << label << "\033[0m " << "Allocating " << size << " bytes, \033[31m" << size / (1024.0 * 1024.0 * 1024.0)
                                  << "\033[0m GB on GPU\n";
        }
    //cudaError_t status = cudaMalloc(devPtr, size);
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, 0);
    cudaError_t status = cudaMallocFromPoolAsync(devPtr, size, pool, 0);
        if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(status) << std::endl;
                exit(1);
    }
    return status;
}

template <typename T>
cudaError_t do_cudaFree_log(T* devPtr, std::string label = "", bool verbose = false) {
        if (verbose) {
                printf("[cudaFree] ");
                print_GPU_info();
                std::cout << "\033[31m" << label << "\033[0m "
                                  << "Freeing GPU memory\n";
        }
    //cudaError_t status = cudaFree(devPtr);
        cudaError_t status = cudaFreeAsync(devPtr, 0);
    if (status != cudaSuccess) {
        std::cerr << "cudaFree failed: " << cudaGetErrorString(status) << std::endl;
                exit(1);
    }
    return status;
}

template <typename A, typename B>
int blocks(A size, B threads) { return (int)((size + threads - 1) / threads); }

void print_last_cuda_error(const char* file, int line, const char* func) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "Cuda error at %s:%d (%s): %s (code %d)\n",
            file, line, func,
            cudaGetErrorString(err), static_cast<int>(err)
        );
    } else {
        printf("No cuda error detected at %s:%d (%s)\n", file, line, func);
    }
}

__global__ void compute_popc(const uint32_t * vector, uint32_t * output, const size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) output[idx] = __popc(vector[idx]);
}

template <typename in_T, typename out_T, typename size_T>
void sum_vector(in_T vector, out_T output, size_t size) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(
        d_temp_storage,
        temp_storage_bytes,
        vector,
        output,
        size
    );
    cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
    cub::DeviceReduce::Sum(
        d_temp_storage,
        temp_storage_bytes,
        vector,
        output,
        size
    );
    cudaFree_log(d_temp_storage, false);
}

template <typename in_T, typename out_T, typename size_T>
void exclusive_sum(in_T vector, out_T output, size_T size) {
    void * d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        vector,
        output,
        size
    );
    cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        vector,
        output,
        size
    );
    cudaFree_log(d_temp_storage, false);
}

template <typename in_T, typename out_T, typename size_T>
void inclusive_sum(in_T vector, out_T output, size_T size) {
    void * d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        vector,
        output,
        size
    );
    cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
    cub::DeviceScan::InclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        vector,
        output,
        size
    );
    cudaFree_log(d_temp_storage, false);
}

template <typename T>
__global__ void finalize_sum(T* output, T* last, size_t size) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        output[size] = output[size-1] + last[0];
}

// See for which cases you REALLY need a complete sum
// Output will be "0,inclusive_sum", so output should have size+1
// Safe for in-place operations
template <typename in_T, typename out_T>
// vector can be an ITERATOR or a pointer. output will always be a pointer
void complete_sum(in_T vector, out_T * output, size_t size) {
    out_T * d_last;
    cudaMalloc_log(&d_last, sizeof(out_T));
    thrust::copy_n(thrust::device, vector + size - 1, 1, d_last); // Works for pointer and also iterator
    exclusive_sum(vector, output, size);
    finalize_sum<<<1,1>>>(output, d_last, size);
    cudaFree_log(d_last);
}

template <typename in_T, typename out_T, typename mask_T>
size_t device_select_flagged(
    in_T d_in,
    out_T d_out,
    mask_T d_mask,
    size_t size
) {
    void * d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    size_t * d_num_selected = nullptr;
    cudaMalloc_log(&d_num_selected, sizeof(size_t), false);

        cudaError_t cuda_status = cub::DeviceSelect::Flagged(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_mask,
        d_out,
        d_num_selected,
        size
    );
    if (cuda_status != cudaSuccess) {
        std::cerr << "error: " << cudaGetErrorString(cuda_status) << std::endl;
                exit(1);
        }
    cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
        cuda_status = cub::DeviceSelect::Flagged(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_mask,
        d_out,
        d_num_selected,
        size
    );
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
__device__ NullableType<A> binary_search(
        A idx_start,
        A idx_end, // exclusive
        B * values,
        C target
) {
        NullableType<A> result;
        while (idx_start < idx_end) {
                A middle_point = idx_start + (idx_end - idx_start) / 2;
                if (values[middle_point] < target) idx_start = middle_point + 1;
                else if (target < values[middle_point]) idx_end = middle_point;
                else {
                        result.value = middle_point;
                        return result;
                }
        }
        result.is_null = true;
        return result;
}

template <typename A, typename B, typename C>
__device__ A binary_search_bounds(
    A idx_start,
    A idx_end, // exclusive (just pass len(values) in many cases)
    B * values,
    C target
) {
    while (idx_start < idx_end) {
        A middle_point = idx_start + (idx_end - idx_start) / 2;
        if (values[middle_point] <= target) idx_start = middle_point + 1;
        else idx_end = middle_point;
    }
    return idx_start - 1;
}

template <typename in_Type, typename idxs_type, typename out_Type>
__global__ void select(in_Type in, idxs_type idxs, out_Type out, size_t size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) out[idx] = in[idxs[idx]];
}

__host__ void read_mtx(const std::string& input_file, std::vector<uint64_t>& cols_rows_zipped) {
    std::ifstream file(input_file);
    if (!file) throw std::runtime_error("cannot open file");
    std::string line;
    std::getline(file, line);  // %%matrixmarket ...
    // skip comments
    while (std::getline(file, line)) if (line[0] != '%') break;

    // read matrix size
    size_t n_rows, n_cols, nnz;
    std::stringstream ss(line);
    ss >> n_rows >> n_cols >> nnz;
        if (n_rows != n_cols) {
                std::ostringstream oss;
                oss << "matrix is not square, sizes are " << n_rows << ", " << n_cols;
                throw std::runtime_error(oss.str());
        }
        N = 32 * ((n_rows + 31) / 32);
        cudaMemcpyToSymbol(d_N, &N, sizeof(uint32_t));

    // read entries
        cols_rows_zipped.reserve(nnz);
    uint32_t r, c;
        double v;
        // write down only l values
    while (file >> r >> c) {
                if (file.peek() != '\n' && file >> v) {};
                r = r-1;
                c = c-1;
                uint64_t zipped;
                if (c < r) {
                        zipped = (uint64_t(c) << 32) | uint64_t(r);
                        cols_rows_zipped.push_back(zipped);
                } else if (r < c) {
                        zipped = (uint64_t(r) << 32) | uint64_t(c);
                        cols_rows_zipped.push_back(zipped);
                }
        }
}

struct Config {
    std::string input_file;
};

std::string get_usage(const char* program_name) {
    return std::string("Usage: ") + program_name + " -i <input_file>\n"
           "Options:\n"
           "  -i <input_file>    Path to the input file (required)\n";
}

Config program_options(int argc, char* argv[]) {
    Config config;

    if (argc == 1) {
        fprintf(stderr, "%s", get_usage(argv[0]).c_str());
        std::exit(EXIT_FAILURE);
    }

    int opt;
    while ((opt = getopt(argc, argv, "i:")) != -1) {
        switch (opt) {
            case 'i':
                config.input_file = optarg;
                break;
            case '?':   // unknown option
            default:
                fprintf(stderr, "Error: unknown option or missing argument\n");
                fprintf(stderr, "%s", get_usage(argv[0]).c_str());
                std::exit(EXIT_FAILURE);
        }
    }

    // Check that required input file was provided
    if (config.input_file.empty()) {
        fprintf(stderr, "Error: input file is required (-i option)\n");
        fprintf(stderr, "%s", get_usage(argv[0]).c_str());
        std::exit(EXIT_FAILURE);
    }
    if (optind < argc) {
        fprintf(stderr, "Warning: extra arguments ignored\n");
    }

    return config;
}

__global__ void find_global_idxs(const uint64_t * d_cols_rows_zipped, uint64_t * global_idxs, const uint32_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) {
                uint32_t c = (uint32_t)(d_cols_rows_zipped[idx] >> 32);
                uint32_t r = (uint32_t)d_cols_rows_zipped[idx];
                global_idxs[idx] = r + (c / 32) * d_N;
        }
}

__global__ void find_global_idxs_u(const uint64_t * d_cols_rows_zipped, uint64_t * global_idxs, const uint32_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) {
                uint32_t c = (uint32_t)(d_cols_rows_zipped[idx] >> 32);
                uint32_t r = (uint32_t)d_cols_rows_zipped[idx];
                global_idxs[idx] = c * (d_N/ 32) + r / 32;
        }
}

template <typename T, typename T_size>
__host__ uint32_t unique(
    T *& d_in,
    T_size num_items
) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

        T * d_out;
    cudaMalloc_log(&d_out, num_items * sizeof(T));
        uint32_t * d_num_selected;
    cudaMalloc_log(&d_num_selected, sizeof(uint32_t));

    cub::DeviceSelect::Unique(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        d_num_selected,
        num_items
    );
    cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
    cub::DeviceSelect::Unique(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        d_num_selected,
        num_items
    );
    cudaFree_log(d_temp_storage);
        cudaFree_log(d_in);
        d_in = d_out;

    uint32_t h_count;
    cudaMemcpy(&h_count, d_num_selected, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree_log(d_num_selected);
        return h_count;
}

template <typename T>
__host__ void radix_sort(T *& input, const size_t size) {
    void * d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    T * output;
    cudaMalloc_log(&output, size * sizeof(T));

    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        input, output,
        size
    );
    cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        input, output,
        size
    );

    cudaFree_log(input);
    cudaFree_log(d_temp_storage);

    input = output;
}

template <typename keyt, typename valuet>
__host__ void radix_sort_pairs(keyt *& global_idxs, valuet *& d_cols, const uint32_t size) {
        void * d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        keyt * global_idxs_out;
        valuet * d_cols_out;
        cudaMalloc_log(&global_idxs_out, size * sizeof(keyt));
        cudaMalloc_log(&d_cols_out, size * sizeof(valuet));

        cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        global_idxs,
        global_idxs_out,
        d_cols,
        d_cols_out,
        size);
    cudaMalloc_log(&d_temp_storage, temp_storage_bytes, false);
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        global_idxs,
        global_idxs_out,
        d_cols,
                d_cols_out,
        size);

        cudaFree_log(global_idxs);
    cudaFree_log(d_cols);
    cudaFree_log(d_temp_storage, false);

        global_idxs = global_idxs_out;
        d_cols = d_cols_out;
}

struct Or_Custom {
        template <typename T>
        __device__ __forceinline__
        T operator() (const T &a, const T &b) const {
                return a | b;
        }
};

struct Popc_Custom {
        template <typename T>
        __device__
        T operator() (const T& a) const {
                return __popc(a);
        }
};

template <typename inputiteratort, typename outputiteratort, typename offsetiteratort>
__host__ void device_segmented_reduce_sum(
        inputiteratort d_input,
        outputiteratort d_output,
        uint32_t num_segments,
        offsetiteratort d_offsets_it
) {
        void * d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceSegmentedReduce::Sum(
                d_temp_storage, temp_storage_bytes,
                d_input,
                d_output,
                num_segments,
                d_offsets_it,
                d_offsets_it + 1);
        cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(
                d_temp_storage, temp_storage_bytes,
                d_input,
                d_output,
                num_segments,
                d_offsets_it,
                d_offsets_it + 1);
        cudaFree_log(d_temp_storage);
}

template <typename keysinputiteratort, typename valuesinputiteratort, typename numitemst, typename reductionopt>
__host__ std::tuple<uint64_t *, uint32_t *, uint32_t> reduce_by_key(
        keysinputiteratort d_keys_in,
        valuesinputiteratort d_values_in,
        numitemst size,
        reductionopt reduction_op)
{
        uint64_t * d_keys_out;
        uint32_t * d_values_out, * d_num_runs_out;
        cudaMalloc_log(&d_keys_out, size * sizeof(uint64_t));
        cudaMalloc_log(&d_values_out, size * sizeof(uint32_t));
        cudaMalloc_log(&d_num_runs_out, sizeof(uint32_t));
        void * d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                d_keys_in,
                d_keys_out,
                d_values_in,
                d_values_out,
                d_num_runs_out,
                reduction_op,
                size);
        cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
        cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                d_keys_in,
                d_keys_out,
                d_values_in,
                d_values_out,
                d_num_runs_out,
                reduction_op,
                size);
    cudaFree_log(d_temp_storage);
        // send d_num_runs out to host
        uint32_t num_runs_out;
        cudaMemcpy(&num_runs_out, d_num_runs_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree_log(d_num_runs_out);
        return {d_keys_out, d_values_out, num_runs_out};
}

template <typename ReductionOpt>
__host__ uint32_t compress_mask(uint64_t *& global_idxs, uint32_t *& d_mask_l, const uint32_t size, ReductionOpt reduction_op) {
        auto [global_idxs_out, d_mask_lout, num_runs_out] = reduce_by_key(global_idxs, d_mask_l, size, reduction_op);

        cudaFree_log(global_idxs);
    cudaFree_log(d_mask_l);
        d_mask_l = d_mask_lout;
        global_idxs = global_idxs_out;

        return num_runs_out;
}

template <typename T>
__global__ void compute_histogram(const T * d_input, uint32_t * d_hist, size_t size) {
    uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
    if (idx < size) atomicAdd(&d_hist[d_input[idx]], 1);
}

// alternative for my histogram:
template <typename T>
__global__ void compute_histogram_l(const T * d_input, uint32_t * d_hist, size_t size) {
    uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
    if (idx < size) atomicAdd(&d_hist[d_input[idx]/d_N], 1);
}

template <typename T>
__global__ void compute_histogram_u(const T * d_input, uint32_t * d_hist, size_t size) {
    uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
    if (idx < size) atomicAdd(&d_hist[d_input[idx]/(d_N/32)], 1);
}

struct divide_by_N {
        template <typename T>
    __device__
    T operator() (const T& x) const {
        return x / T(d_N);
    }
};

struct divide_by_N_32 {
        template <typename T>
    __device__
    T operator() (const T& x) const {
        return x / T(d_N/32);
    }
};

template <typename inT, typename outT>
__global__ void modulo_N(const inT * in, outT * out, const uint32_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) out[idx] = in[idx] % d_N;
}

template <typename inT, typename outT>
__global__ void modulo_N_32(const inT * in, outT * out, const uint32_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) out[idx] = in[idx] % (d_N/32);
}

__global__ void get_rows(const uint64_t * d_cols_rows_zipped, uint32_t * d_out, size_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) d_out[idx] = uint32_t(d_cols_rows_zipped[idx]);
}

__inline__ __device__ uint32_t warpSum32(uint32_t val) {
    // full warp reduction
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val; // thread 0 has total sum
}

template <typename T>
__global__ void compute_binary_popcount_mma_9(
        const uint32_t * d_mask_u,
        T * d_col_id_u,
        T * d_set_ptr_u,
        const uint32_t * d_mask_l,
        T * d_col_id_l,
        T * d_set_ptr_l,
        T * d_idxs,
        uint64_t * d_result,
    const uint32_t * popc_array,
        const uint64_t offset,
        uint32_t compressed_size_u,
        uint64_t * d_n_blocks_per_row,
        uint32_t non_zero_rows,
        uint64_t total_num_warps,
        uint32_t * d_i_values
) {
        uint64_t real_idx = 4 * (uint64_t) blockIdx.x + threadIdx.y + offset;
        uint64_t idx;
        if (real_idx >= total_num_warps) idx = total_num_warps - 1;
        else idx = real_idx;

        uint32_t row = binary_search_bounds(0u, compressed_size_u, d_n_blocks_per_row, idx);
        uint32_t i = d_i_values[row];
        uint64_t block_dim_x = (popc_array[i+1] - popc_array[i] + 31) / 32;
        uint64_t block_x = block_dim_x - (d_n_blocks_per_row[row+1] - idx);

    __align__(16) __shared__ uint32_t dA[32];
    __align__(16) __shared__ uint32_t dB[4*32];
    __align__(16) __shared__ int32_t dD[4*64];

    // Fill dA
        size_t col = row;
        if (threadIdx.x < 8 && threadIdx.y < 4)
                dA[4 * threadIdx.x + threadIdx.y] = (threadIdx.x == 0 && real_idx == idx) ? d_mask_u[col] : 0;

        // Fill dB
        if (threadIdx.x < 32 && threadIdx.y < 4) {
                uint32_t padding = ((32 - ((popc_array[i+1] - popc_array[i]) & 31)) & 31);
                size_t idxx = popc_array[i] + 32 * block_x + threadIdx.x - padding;
                size_t unrolled_id = 32 * (threadIdx.x / 8) + 4 * (threadIdx.x % 8) + threadIdx.y;
                if (popc_array[i] <= idxx && idxx < popc_array[i+1] && real_idx == idx) {
                        int id = d_idxs[idxx];
                        int c = d_col_id_u[col];
                        NullableType<T> result = binary_search(d_set_ptr_l[c], d_set_ptr_l[c+1], d_col_id_l, id);
                        dB[unrolled_id] = !result.is_null ? d_mask_l[result.value] : 0;
                } else dB[unrolled_id] = 0;
                __syncthreads();
        }

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
        : "l"(dA),
                  "l"(dB + 32 * threadIdx.y),
                  "l"(dD + 64 * threadIdx.y)
        : "memory"
    );
        __syncthreads();

        if (threadIdx.y == 0) {
                uint32_t val = dD[threadIdx.x % 8 + (threadIdx.x / 8) * 64];
                val = warpSum32(val);
                if (threadIdx.x == 0) {
                        atomicAdd((unsigned long long int *)d_result, (unsigned long long int)val);
                }
        }
}

__global__ void create_non_compact_mask_l(const uint64_t * d_cols_rows_zipped, uint32_t * d_mask_l, const uint32_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) {
                uint32_t c = uint32_t(d_cols_rows_zipped[idx] >> 32);
                d_mask_l[idx] = uint32_t(1) << (31 - c % 32);
        }
}

__global__ void create_non_compact_mask_u(const uint64_t * d_cols_rows_zipped, uint32_t * d_mask_u, const uint32_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) {
                uint32_t r = uint32_t(d_cols_rows_zipped[idx]);
                d_mask_u[idx] = uint32_t(1) << (31 - r % 32);
        }
}

template <typename out_T>
__global__ void flag_non_singletons(const uint64_t * d_in , out_T * d_mask, const uint64_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) {
                if (idx == 0) d_mask[idx] = (d_in[idx] >> 32) == (d_in[idx+1] >> 32);
                else if (idx == (size - 1)) d_mask[idx] = (d_in[idx-1] >> 32) == (d_in[idx] >> 32);
                else d_mask[idx] = ((d_in[idx-1] >> 32) == (d_in[idx] >> 32)) || ((d_in[idx] >> 32) == (d_in[idx+1] >> 32));
        }
}

template <typename T>
__host__ uint32_t run_length_encode(T d_input, uint32_t *& d_counts_out, const uint32_t size) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    uint32_t* d_unique_temp;
    uint32_t* d_num_runs_temp;
    cudaMalloc_log(&d_unique_temp, size * sizeof(uint32_t));
    cudaMalloc_log(&d_num_runs_temp, sizeof(uint32_t));
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage,
        temp_storage_bytes,
        d_input,
        d_unique_temp,
        d_counts_out,
        d_num_runs_temp,
        size
    );
    cudaMalloc_log(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage,
        temp_storage_bytes,
        d_input,
        d_unique_temp,
        d_counts_out,
        d_num_runs_temp,
        size
    );
        uint32_t num_runs;
    cudaMemcpy(&num_runs, d_num_runs_temp, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree_log(d_temp_storage);
        cudaFree_log(d_unique_temp);
    cudaFree_log(d_num_runs_temp);
        return num_runs;
}

__global__ void gather_popc_array(uint32_t * popc_array, uint64_t * popc_long, uint32_t * d_set_ptr_u, uint32_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) {
                popc_array[idx] = (uint32_t) popc_long[d_set_ptr_u[idx]];
        }
}

__global__ void compute_n_blocks_per_row(uint64_t * popc_long, uint32_t * d_set_ptr_u, uint32_t * popc_array, uint32_t non_zero_rows, uint32_t * d_i_values, uint32_t size) {
        uint64_t idx = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;
        if (idx < size) {
                //uint32_t i = (uint32_t) (global_idxs[idx] / (d_N/32)); // This would work if zero-rows were not removed
                uint32_t i = binary_search_bounds(0u, non_zero_rows, d_set_ptr_u, idx);
                d_i_values[idx] = i;
                popc_long[idx] = ((uint64_t) popc_array[i+1] - (uint64_t) popc_long[idx] + 31ULL) / 32ULL;
        }
}

uint32_t count_triangles(std::vector<uint64_t> cols_rows_zipped, uint32_t nnz, Config config, bool print_info) {
        uint64_t * d_cols_rows_zipped;
        cudaMalloc_log(&d_cols_rows_zipped, nnz * sizeof(uint64_t));
        cudaMemcpy(d_cols_rows_zipped, cols_rows_zipped.data(), nnz * sizeof(uint64_t), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        radix_sort(d_cols_rows_zipped, nnz);
        nnz = unique(d_cols_rows_zipped, nnz);

        uint64_t * global_idxs;
        cudaMalloc_log(&global_idxs, nnz * sizeof(uint64_t));


        // -------- MATRIX L -------- //
        uint32_t * d_mask_l;
        cudaMalloc_log(&d_mask_l, nnz * sizeof(uint32_t));
        create_non_compact_mask_l<<<blocks(nnz,256),256>>>(d_cols_rows_zipped, d_mask_l, nnz);
        // create global_idxs
        find_global_idxs<<<blocks(nnz,256),256>>>(d_cols_rows_zipped, global_idxs, nnz);
        // sort d_mask_l using global_idxs
        radix_sort_pairs(global_idxs, d_mask_l, nnz);
        // compress elements of mask_d_l_ with same global_idx using or operation
        uint32_t compressed_size_l = 0;
        compressed_size_l = compress_mask(global_idxs, d_mask_l, nnz, Or_Custom{});
        // create d_row_id_l
        uint32_t * d_row_id_l;
        cudaMalloc_log(&d_row_id_l, compressed_size_l * sizeof(uint32_t));
        modulo_N<<<blocks(compressed_size_l,256),256>>>(global_idxs, d_row_id_l, compressed_size_l); // compressed rows
        // create d_set_ptr_l
        uint32_t * d_set_ptr_l;
        cudaMalloc_log(&d_set_ptr_l, (N/32+1) * sizeof(uint32_t));
        cudaMemset(d_set_ptr_l, 0, (N/32+1)*sizeof(uint32_t));
        compute_histogram_l<<<blocks(compressed_size_l,256),256>>>(
                //thrust::make_transform_iterator(global_idxs, divide_by_n()), // compressed columns (no need to store them)
                global_idxs,
                d_set_ptr_l,
                compressed_size_l
        );
        inclusive_sum(d_set_ptr_l, d_set_ptr_l+1, N/32);
        cudaMemset(d_set_ptr_l, 0, sizeof(uint32_t));
        // -------- MATRIX L -------- //

        // PURGE ROWS WITH EXACTLY ONE 1
        uint8_t * d_mask;
        cudaMalloc_log(&d_mask, nnz * sizeof(uint8_t));
        flag_non_singletons<<<blocks(nnz,256),256>>>(d_cols_rows_zipped, d_mask, nnz);
        uint64_t * d_cols_rows_zipped_short;
        cudaMalloc_log(&d_cols_rows_zipped_short, nnz * sizeof(uint64_t));
        nnz = device_select_flagged(d_cols_rows_zipped, d_cols_rows_zipped_short, d_mask, nnz);
        cudaFree_log(d_mask);
        cudaFree_log(d_cols_rows_zipped);
        d_cols_rows_zipped = d_cols_rows_zipped_short;

        // -------- MATRIX U -------- //
        // Create d_mask_u
        uint32_t * d_mask_u;
        cudaMalloc_log(&d_mask_u, nnz * sizeof(uint32_t));
        create_non_compact_mask_u<<<blocks(nnz,256),256>>>(d_cols_rows_zipped, d_mask_u, nnz);
        // Create d_idxs
        uint32_t * d_idxs;
        cudaMalloc_log(&d_idxs, nnz * sizeof(uint32_t));
        get_rows<<<blocks(nnz,256),256>>>(d_cols_rows_zipped, d_idxs, nnz);
        // Create global_idxs_u
        find_global_idxs_u<<<blocks(nnz,256),256>>>(d_cols_rows_zipped, global_idxs, nnz);
        cudaFree_log(d_cols_rows_zipped);
        // Sort d_mask_u using global_idxs
        radix_sort_pairs(global_idxs, d_mask_u, nnz);
        // Compress elements of mask_d_u_ with SAME global_idx using OR operation
        uint32_t compressed_size_u = 0;
        compressed_size_u = compress_mask(global_idxs, d_mask_u, nnz, Or_Custom{});
        // Create d_col_id_u
        uint32_t * d_col_id_u;
        cudaMalloc_log(&d_col_id_u, compressed_size_u * sizeof(uint32_t));
        modulo_N_32<<<blocks(compressed_size_u,256),256>>>(global_idxs, d_col_id_u, compressed_size_u); // Compressed cols
        // Create d_set_ptr_u
        uint32_t * d_set_ptr_u;
        cudaMalloc_log(&d_set_ptr_u, N * sizeof(uint32_t));
        //cudaMemset(d_set_ptr_u, 0, N * sizeof(uint32_t));
        uint32_t non_zero_rows = run_length_encode( // We use RLE instead of histogram to ignore ZEROs in rows (which are completely unnecessary)
                thrust::make_transform_iterator(global_idxs, divide_by_N_32()), // Compressed rows (no need to store them)
                d_set_ptr_u,
                compressed_size_u
        );
        cudaFree_log(global_idxs);
        complete_sum(d_set_ptr_u, d_set_ptr_u, non_zero_rows);
        // -------- MATRIX U -------- //

        uint64_t * popc_long;
        cudaMalloc_log(&popc_long, (compressed_size_u + 1) * sizeof(uint64_t));
        complete_sum(
                thrust::make_transform_iterator(d_mask_u, Popc_Custom()),
                popc_long,
                compressed_size_u
        );
        // Create popc_array
        uint32_t * popc_array;
        cudaMalloc_log(&popc_array, (non_zero_rows+1) * sizeof(uint32_t));
        gather_popc_array<<<blocks(non_zero_rows + 1, 256),256>>>(popc_array, popc_long, d_set_ptr_u, non_zero_rows + 1);
        // Create d_i_values
        uint32_t * d_i_values;
        cudaMalloc_log(&d_i_values, compressed_size_u * sizeof(uint32_t));
        compute_n_blocks_per_row<<<blocks(compressed_size_u,256),256>>>(popc_long, d_set_ptr_u, popc_array, non_zero_rows, d_i_values, compressed_size_u);
        complete_sum(
                popc_long,
                popc_long,
                compressed_size_u
        );
        uint64_t * d_n_blocks_per_row = popc_long;
        uint64_t total_num_warps;
        cudaMemcpy(&total_num_warps, d_n_blocks_per_row + compressed_size_u, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        uint64_t total_num_blocks = (total_num_warps + 3ULL) / 4ULL; // A block will be 4 warps

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_preprocessing = 0.0f;
        cudaEventElapsedTime(&ms_preprocessing, start, stop);

        uint64_t triangle_count = 0, * d_result;
        cudaMalloc_log(&d_result, sizeof(uint64_t));
        cudaMemset(d_result, 0, sizeof(uint64_t));

        uint64_t offset = 0ULL;
        int blocksPerGrid = 0x7fffffff;
        dim3 threadsPerBlock(32, 4);
        uint32_t number_of_launches = (uint32_t)((total_num_blocks + 0x7fffffffULL - 1) / 0x7fffffffULL);
        if (number_of_launches > 1)
                throw std::logic_error("Not implemented: total_num_blocks is bigger than INTMAX_32");

        cudaEventRecord(start);
    for (uint32_t launch = 0; launch < number_of_launches; ++launch) {
        if (launch == number_of_launches - 1) blocksPerGrid = total_num_blocks % 0x7fffffffULL;
        compute_binary_popcount_mma_9 <<<blocksPerGrid, threadsPerBlock>>>
                        (d_mask_u,
                        d_col_id_u,
                        d_set_ptr_u,
                        d_mask_l,
                        d_row_id_l,
                        d_set_ptr_l,
                        d_idxs,
                        d_result,
                        popc_array,
                        offset,
                        compressed_size_u,
                        d_n_blocks_per_row,
                        non_zero_rows,
                        total_num_warps,
                        d_i_values);
                offset += blocksPerGrid;
    }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree_log(d_mask_u);
        cudaFree_log(d_col_id_u);
        cudaFree_log(d_set_ptr_u);
        cudaFree_log(popc_array);
        cudaFree_log(d_idxs);
        cudaFree_log(d_mask_l);
        cudaFree_log(d_row_id_l);
        cudaFree_log(d_set_ptr_l);
        cudaFree_log(d_n_blocks_per_row);
        cudaFree_log(d_i_values);

    cudaMemcpy(&triangle_count, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree_log(d_result);
        cudaDeviceSynchronize();

        // Print veredict
        if (print_info) {
                printf(
                        "============================================================\n"
                        "                 Triangle Counting Summary\n"
                        "============================================================\n"
                        "Graph              : %s\n"
                        "Blocks of threads  : %zu\n"
                        "Preprocessing (ms) : %.3f\n"
                        "Kernel (ms)        : %.3f\n"
                        "Triangles          : %llu\n"
                        "============================================================\n",
                        config.input_file.c_str(),
                        total_num_blocks,
                        ms_preprocessing,
                        ms,
                        (unsigned long long)triangle_count
                );
                print_last_cuda_error(__FILE__, __LINE__, __func__);
        }

        return 0;
}

void allocate_memory(void) {
        cudaMemPool_t mempool;
        cudaDeviceGetDefaultMemPool(&mempool, 0);
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(
                mempool,
                cudaMemPoolAttrReleaseThreshold,
                &threshold
        );
        // Allocate all GPU into the memory pool
        uint8_t * ptr;
        size_t free_byte, total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);
        // Leave safety margin (5% is enough, usually)
        size_t to_allocate = free_byte * 0.95;
        cudaMalloc_log(&ptr, to_allocate);
        cudaFree_log(ptr);
}


int main(int argc, char* argv[]) {
        std::cout << "Thrust version: "
                  << THRUST_MAJOR_VERSION << "."
                  << THRUST_MINOR_VERSION << "."
                  << THRUST_SUBMINOR_VERSION << std::endl;
        std::cout << "CUB version: " << CUB_VERSION << std::endl;

        cudaFree(0); // Force creation of context
        cudaDeviceSynchronize();
        allocate_memory();

        Config config = program_options(argc, argv);
        std::string arg = argv[1];

        std::cout << "Reading data from " << config.input_file << "...\n";
        std::vector<uint64_t> cols_rows_zipped;
        read_mtx(config.input_file, cols_rows_zipped);
        uint32_t nnz = cols_rows_zipped.size();

        // Warmup
        uint64_t traingle_count = count_triangles(cols_rows_zipped, nnz, config, false);

        // Real
        traingle_count = count_triangles(cols_rows_zipped, nnz, config, true);

        return 0;
}
