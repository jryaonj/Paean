#include "util.h"
#include "cuda_util.h"
#include "cub.cuh"
#include "cub_sort.cuh"
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


void cubRadixSortKey(uint64_t *d_keys_in, uint64_t *d_keys_out, 
                     uint32_t numOfEntry)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                   d_keys_in, d_keys_out, numOfEntry);
    // allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                   d_keys_in, d_keys_out, numOfEntry);
}

void cubRadixSortInterval(d_Gaps &d_intervals_in, d_Gaps &d_intervals_out, 
                          uint32_t numOfInterval)
{
    // with junctions
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_intervals_in.start_, d_intervals_out.start_,
                                    d_intervals_in.end_, d_intervals_out.end_,
                                    numOfInterval);
    // allocate temporary storage
    CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_intervals_in.start_, d_intervals_out.start_,
                                    d_intervals_in.end_, d_intervals_out.end_,
                                    numOfInterval);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void cubRadixSortJunction(d_Junctions &d_junctions_in, d_Junctions &d_junctions_out,
                             h_Junctions &h_junctions, uint32_t numOfJunction)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                   d_junctions_in.end_, d_junctions_out.end_,
                                   d_junctions_in.start_, d_junctions_out.start_,
                                   numOfJunction);
    // allocate temporary storage
    CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                   d_junctions_in.end_, d_junctions_out.end_,
                                   d_junctions_in.start_, d_junctions_out.start_,
                                   numOfJunction);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    // determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_junctions_out.start_, d_junctions_in.start_,
                                    d_junctions_out.end_, d_junctions_in.end_,
                                    numOfJunction);
    // allocate temporary storage
    CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_junctions_out.start_, d_junctions_in.start_,
                                    d_junctions_out.end_, d_junctions_in.end_,
                                    numOfJunction);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUDA_SAFE_CALL(cudaMemcpy(d_junctions_out.start_, d_junctions_in.start_,
                              sizeof(uint64_t) * numOfJunction,
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_junctions_out.end_, d_junctions_in.end_,
                              sizeof(uint64_t) * numOfJunction,
                              cudaMemcpyDeviceToDevice));
}


d_Junctions thrustSegmentedScanJunction(d_Junctions &d_junctions_in, uint32_t &numOfJunction)
{
    // segmented prefix sum
    thrust::device_vector<uint32_t> d_counts_start(numOfJunction);
    thrust::device_vector<uint32_t> d_counts_end(numOfJunction);
    thrust::fill(thrust::device, d_counts_start.begin(), d_counts_start.end(), 1);
    thrust::fill(thrust::device, d_counts_end.begin(), d_counts_end.end(), 1);
    thrust::inclusive_scan_by_key(thrust::device, d_junctions_in.start_,
                                  d_junctions_in.start_ + numOfJunction,
                                  d_counts_start.begin(), d_counts_start.begin());
    thrust::inclusive_scan_by_key(thrust::device, d_junctions_in.end_,
                                  d_junctions_in.end_ + numOfJunction,
                                  d_counts_end.begin(), d_counts_end.begin());
    
    thrust::device_vector<uint32_t> d_counts(numOfJunction);
    thrust::transform(thrust::device, d_counts_start.begin(), d_counts_start.end(),
                      d_counts_end.begin(), d_counts.begin(), min_element<uint32_t>());

    // set flags
    thrust::device_vector<uint32_t> d_flags(numOfJunction);
    thrust::replace_copy_if(thrust::device, d_counts.begin(), d_counts.end(),
                            d_flags.begin(), is_greater_than_one<uint32_t>(), 0);

    // compute offsets
    thrust::device_vector<uint32_t> d_indices(numOfJunction);
    thrust::inclusive_scan(thrust::device, d_flags.begin(),
                           d_flags.end(), d_indices.begin());

    // calculate new numOfJunction
    uint32_t new_numOfJunction;
    CUDA_SAFE_CALL(cudaMemcpy(&new_numOfJunction,
                              thrust::raw_pointer_cast(d_indices.data()) + numOfJunction - 1,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

    d_Junctions d_junctions_out;
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&d_junctions_out.start_, sizeof(uint64_t) * new_numOfJunction));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&d_junctions_out.end_, sizeof(uint64_t) * new_numOfJunction));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&d_junctions_out.count, sizeof(uint32_t) * new_numOfJunction));

    // compute number of junction block
    unsigned nJunctionBlock = ceilDiv(numOfJunction, blockSize);

    scatter_if<uint64_t><<<nJunctionBlock, blockSize>>>(
                        thrust::raw_pointer_cast(d_indices.data()),
                        d_junctions_in.start_, d_junctions_out.start_,
                        thrust::raw_pointer_cast(d_flags.data()),
                         numOfJunction);
    scatter_if<uint64_t><<<nJunctionBlock, blockSize>>>(
                        thrust::raw_pointer_cast(d_indices.data()),
                        d_junctions_in.end_, d_junctions_out.end_,
                        thrust::raw_pointer_cast(d_flags.data()),
                        numOfJunction);
    scatter_if<uint32_t><<<nJunctionBlock, blockSize>>>(
                        thrust::raw_pointer_cast(d_indices.data()),
                        thrust::raw_pointer_cast(d_counts.data()),
                        d_junctions_out.count,
                        thrust::raw_pointer_cast(d_flags.data()),
                        numOfJunction);
    // update numOfJunction
    numOfJunction = new_numOfJunction;

    return d_junctions_out;
}

// sort bins by using cub library
void cubRadixSortBin(d_Bins &d_bins_in, d_Bins &d_bins_out, 
                      h_Bins &h_bins, uint32_t numOfBin)
{
    // indices on cpu
    auto *indices = new uint32_t[numOfBin];
    std::iota(indices, indices+numOfBin, 0);

    // indices on gpu
    uint32_t *d_indices_in;
    uint32_t *d_indices_out;
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&d_indices_in, sizeof(uint32_t) * numOfBin));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&d_indices_out, sizeof(uint32_t) * numOfBin));
    CUDA_SAFE_CALL(
        cudaMemcpy(d_indices_in, indices, sizeof(uint32_t) * numOfBin,
                   cudaMemcpyHostToDevice));

    // with junctions
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    //! determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_bins_in.start_, d_bins_out.start_,
                                    d_indices_in, d_indices_out, numOfBin);
    // allocate temporary storage
    CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_bins_in.start_, d_bins_out.start_,
                                    d_indices_in, d_indices_out, numOfBin);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // compute number of thread block for bins
    unsigned nBinBlock = ceilDiv(numOfBin, blockSize);

    gather<uint64_t><<<nBinBlock, blockSize>>>(d_indices_out, d_bins_in.end_,
                                               d_bins_out.end_, numOfBin);
    gather<uint8_t><<<nBinBlock, blockSize>>>(d_indices_out, d_bins_in.strand,
                                              d_bins_out.strand, numOfBin);
    gather<bin_core_t><<<nBinBlock, blockSize>>>(d_indices_out, d_bins_in.core,
                                                 d_bins_out.core, numOfBin);
    CUDA_SAFE_CALL(cudaMemcpy(h_bins.start_.data(), d_bins_out.start_, 
                        numOfBin * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_bins.end_.data(), d_bins_out.end_, 
                        numOfBin * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_bins.strand.data(), d_bins_out.strand,
                        numOfBin * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_bins.core.data(), d_bins_out.core,
                        numOfBin * sizeof(bin_core_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // free used memory
    delete []indices;
    CUDA_SAFE_CALL(cudaFree(d_indices_in));
    CUDA_SAFE_CALL(cudaFree(d_indices_out));
}

// sort ases by using cub library
void cubRadixSortASE(d_ASEs &d_ases_in, d_ASEs &d_ases_out,
                     uint32_t numOfASE)
{
    // indices on cpu
    auto *indices = new uint32_t[numOfASE];
    std::iota(indices, indices+numOfASE, 0);

    // indices on gpu
    uint32_t *d_indices_in;
    uint32_t *d_indices_out;
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&d_indices_in, sizeof(uint32_t) * numOfASE));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&d_indices_out, sizeof(uint32_t) * numOfASE));
    CUDA_SAFE_CALL(
        cudaMemcpy(d_indices_in, indices, sizeof(uint32_t) * numOfASE,
                   cudaMemcpyHostToDevice));

    // with junctions
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    //! determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_ases_in.start_, d_ases_out.start_,
                                    d_indices_in, d_indices_out, numOfASE);
    // allocate temporary storage
    CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_ases_in.start_, d_ases_out.start_,
                                    d_indices_in, d_indices_out, numOfASE);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // compute number of thread block for ases
    unsigned nASEBlock = ceilDiv(numOfASE, blockSize);

    gather<uint64_t><<<nASEBlock, blockSize>>>(d_indices_out, d_ases_in.end_,
                                               d_ases_out.end_, numOfASE);
    gather<uint8_t><<<nASEBlock, blockSize>>>(d_indices_out, d_ases_in.strand,
                                              d_ases_out.strand, numOfASE);
    gather<ase_core_t><<<nASEBlock, blockSize>>>(d_indices_out, d_ases_in.core,
                                                 d_ases_out.core, numOfASE);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // free used memory
    delete []indices;
    CUDA_SAFE_CALL(cudaFree(d_indices_in));
    CUDA_SAFE_CALL(cudaFree(d_indices_out));
}

// cub reduce sum
void cubReduceSum(float *d_in, float *d_out, uint32_t num_items)
{
    // determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int num_items_ = int(num_items);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                           d_in, d_out, num_items_);
    // allocate temporary storage
    CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                           d_in, d_out, num_items_);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
