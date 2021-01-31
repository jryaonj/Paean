#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>

#include <stdio.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>

#include "bin.h"
#include "parse.h"
#include "fusion.h"
#include "util.h"
#include "cuda_util.h"

#include "cub_sort.cuh"
#include "bin_assign_kernel.cuh"
#include "bin_calc_kernel.cuh"

// log file
std::ofstream log_file;

d_Reads gpu_MallocRead(h_Reads &h_reads, uint32_t numOfRead, bool copy=true)
{
    d_Reads d_reads;
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_reads.start_), sizeof(uint64_t) * numOfRead));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_reads.end_), sizeof(uint64_t) * numOfRead));
    // CUDA_SAFE_CALL(
    //     cudaMalloc((void **)&(d_reads.strand), sizeof(uint8_t) * numOfRead));

    if (!copy) return d_reads;

    CUDA_SAFE_CALL(cudaMemcpy(d_reads.start_, h_reads.start_.data(),
                              sizeof(uint64_t) * numOfRead,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_reads.end_, h_reads.end_.data(),
                              sizeof(uint64_t) * numOfRead,
                              cudaMemcpyHostToDevice));
    // CUDA_SAFE_CALL(cudaMemcpy(d_reads.strand, h_reads.strand.data(),
    //                           sizeof(uint8_t) * numOfRead,
    //                           cudaMemcpyHostToDevice));

    return d_reads;
}

d_Junctions gpu_MallocJunction(h_Junctions &h_junctions, uint32_t numOfJunction, bool copy=true)
{
    d_Junctions d_junctions;
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_junctions.start_), sizeof(uint64_t) * numOfJunction));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_junctions.end_), sizeof(uint64_t) * numOfJunction));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_junctions.count), sizeof(uint32_t) * numOfJunction));

    if (!copy) return d_junctions;

    CUDA_SAFE_CALL(cudaMemcpy(d_junctions.start_, h_junctions.start_.data(),
                              sizeof(uint64_t) * numOfJunction,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_junctions.end_, h_junctions.end_.data(),
                              sizeof(uint64_t) * numOfJunction,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemset(d_junctions.count, 0, sizeof(uint32_t) * numOfJunction));

    return d_junctions;
}

d_Gaps gpu_MallocGap(h_Gaps &h_gaps, uint32_t numOfGap, bool copy=true)
{
    d_Gaps d_gaps;
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_gaps.start_), sizeof(uint64_t) * numOfGap));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_gaps.end_), sizeof(uint64_t) * numOfGap));

    if (!copy) return d_gaps;

    CUDA_SAFE_CALL(cudaMemcpy(d_gaps.start_, h_gaps.start_.data(),
                              sizeof(uint64_t) * numOfGap,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_gaps.end_, h_gaps.end_.data(),
                              sizeof(uint64_t) * numOfGap,
                              cudaMemcpyHostToDevice));

    return d_gaps;
}

d_ASEs gpu_MallocASE(h_ASEs &h_ases, uint32_t numOfASE, bool copy=true)
{
    d_ASEs d_ases;
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_ases.start_), sizeof(uint64_t) * numOfASE));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_ases.end_), sizeof(uint64_t) * numOfASE));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_ases.strand), sizeof(uint8_t) * numOfASE));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_ases.core), sizeof(ase_core_t) * numOfASE));

    if (!copy) return d_ases;

    CUDA_SAFE_CALL(cudaMemcpy(d_ases.start_, h_ases.start_.data(),
                              sizeof(uint64_t) * numOfASE,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_ases.end_, h_ases.end_.data(),
                              sizeof(uint64_t) * numOfASE,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_ases.strand, h_ases.strand.data(),
                              sizeof(uint8_t) * numOfASE,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_ases.core, h_ases.core.data(),
                              sizeof(ase_core_t) * numOfASE,
                              cudaMemcpyHostToDevice));

    return d_ases;
}

d_Bins gpu_MallocBin(h_Bins &h_bins, uint32_t numOfBin, bool copy=true)
{
    d_Bins d_bins;
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_bins.start_), sizeof(uint64_t) * numOfBin));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_bins.end_), sizeof(uint64_t) * numOfBin));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_bins.strand), sizeof(uint8_t) * numOfBin));
    CUDA_SAFE_CALL(
        cudaMalloc((void **)&(d_bins.core), sizeof(bin_core_t) * numOfBin));

    if (!copy) return d_bins;

    CUDA_SAFE_CALL(cudaMemcpy(d_bins.start_, h_bins.start_.data(),
                              sizeof(uint64_t) * numOfBin,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_bins.end_, h_bins.end_.data(),
                              sizeof(uint64_t) * numOfBin,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_bins.strand, h_bins.strand.data(),
                              sizeof(uint8_t) * numOfBin,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_bins.core, h_bins.core.data(),
                              sizeof(bin_core_t) * numOfBin,
                              cudaMemcpyHostToDevice));

    return d_bins;
}

void output_tpm(h_Bins &h_bins, float *tpmStore,
                float tpmCount, uint32_t *readCount,
                uint32_t *nj_readCount, uint32_t *all_readCount,
                std::string &output_dir, std::string &file_prefix)
{
    std::string bin_file = pathJoin(output_dir,
                                    file_prefix + "_gene.tsv");
    std::ofstream fout(bin_file);
    fout.setf(std::ios::fixed);

    // tpm
    fout << "gene_id\t"
         << "gene_name\t"
         << "reference\t"
         << "strand\t"
         << "reads_with_junction\t"
         << "reads_without_junction\t"
         << "total_reads\t"
         << "tpm\n";
    uint32_t size_ = h_bins.core.size();
    std::vector<std::string> attrs;
    for (uint32_t j = 0; j < size_; j++) {
        auto bin = bin_gid_map.find(h_bins.core[j].gid_h);
        assert(bin != bin_gid_map.end());
        attrs = split(bin->second);
        fout << attrs[0] << "\t"
             << attrs[1] << "\t"
             << attrs[2] << "\t"
             << attrs[3] << "\t"
             << readCount[j] << "\t"
             << nj_readCount[j] << "\t"
             << all_readCount[j] << "\t"
             << std::setprecision(3) << tpmStore[j] << "\n";
    }
    // fout << std::setprecision(3) << tpmCount << "\n";
    fout.close();
}

void output_psi(ASEPsi *h_ase_psi, uint32_t numOfASE,
                uint32_t ase_type, std::string &output_dir,
                std::string &file_prefix)
{
    std::string ase_file = pathJoin(output_dir, 
                            file_prefix + "_ase_" + 
                            aseTypes[ase_type] + ".tsv");
    std::ofstream fout(ase_file);
    fout.setf(std::ios::fixed);

    // print ASE type
    // fout << "ASE Type: " << aseTypes[ase_type] << "\n";

    // print header
    fout << "ase_id\t"
         << "gene_name\t"
         << "count_in\t"
         << "count_out\t"
         << "psi\t"
         << "psi_lb\t"
         << "psi_ub\n";

    // uint32_t countIn, countOut;
    // countIn = countOut = 0;

    for (uint32_t i = 0; i < numOfASE; i++) {
        // if (h_ase_psi[i].countIn) countIn++;
        // if (h_ase_psi[i].countOut) countOut++;

        auto gid = ase_gid_map.find(h_ase_psi[i].gid_h);
        fout << gid->second << "\t";

        std::stringstream ss;
        uint32_t binCount = h_ase_psi[i].bin_h.binCount;
        if (binCount) {
            std::vector<std::string> attrs;
            for (uint32_t j = 0; j < binCount; j++) {
                auto bin = bin_gid_map.find(h_ase_psi[i].bin_h.bins[j]);
                if (bin == bin_gid_map.end()) {
                    ss << "null";
                } else {
                    attrs = split(bin->second);
                    ss << attrs[1];
                }
                if (j != (binCount - 1))
                    ss << ",";
            }
        } else {
            ss << "null";
        }
        fout << ss.str() << "\t";
        fout << std::setprecision(3) << h_ase_psi[i].countIn << "\t"
             << h_ase_psi[i].countOut << "\t"
             << h_ase_psi[i].psi << "\t"
             << h_ase_psi[i].ciStart << "\t"
             << h_ase_psi[i].ciEnd << "\n";
    }
    // fout << std::setprecision(3) << countIn << "\t"
    //      << std::setprecision(3) << countOut << "\n";
    fout.close();
}

void output_fusion(std::string &output_dir, std::string &file_prefix)
{
    std::string fusion_file = pathJoin(output_dir, 
                                       file_prefix + "_fusion.tsv");
    std::ofstream fout(fusion_file);

    fout << "gene_A\t" 
         << "gene_B\t" 
         << "read_count\n";
    std::vector<std::string> attrsA, attrsB;
    for (auto &bin : bin_read_map) {
        auto binA = bin_gid_map.find(bin.first.first);
        auto binB = bin_gid_map.find(bin.first.second);
        assert(binA != bin_gid_map.end() && 
               binB != bin_gid_map.end());
        attrsA = split(binA->second);
        attrsB = split(binB->second);
        fout << attrsA[1] << "\t"
             << attrsB[1] << "\t"
             << bin.second << "\n";
    }
    fout.close();
}

void Run_tpm(d_Reads d_reads, d_Reads d_nj_reads,
             h_Bins &h_bins, h_Gaps &h_gaps,
             uint32_t numOfRead, uint32_t numOf_nj_Read,
             uint32_t numOfBin, int mode,
             std::string &output_dir, std::string &file_prefix)
{
    // allocate and copy bins to device global memory
    // std::cout << "allocate gpu memory for bins..." << std::endl;
    d_Bins d_unsorted_bins = gpu_MallocBin(h_bins, numOfBin);
    d_Bins d_bins = gpu_MallocBin(h_bins, numOfBin, false);

    // std::cout << "starting sorting bins..." << std::endl;
    // sort bins
    cubRadixSortBin(d_unsorted_bins, d_bins, h_bins, numOfBin);

    // free memory
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.start_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.end_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.strand));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.core));

    // record read counts
    uint32_t *d_readCount, *d_nj_readCount, *d_all_readCount;
    size_t bytes_i = sizeof(uint32_t) * numOfBin;
    CUDA_SAFE_CALL(cudaMalloc((void **)&(d_readCount), bytes_i));
    CUDA_SAFE_CALL(cudaMemset(d_readCount, 0, bytes_i));
    CUDA_SAFE_CALL(cudaMalloc((void **)&(d_nj_readCount), bytes_i));
    CUDA_SAFE_CALL(cudaMemset(d_nj_readCount, 0, bytes_i));
    CUDA_SAFE_CALL(cudaMalloc((void **)&(d_all_readCount), bytes_i));
    CUDA_SAFE_CALL(cudaMemset(d_all_readCount, 0, bytes_i));

    // compute number of thread block
    unsigned nBinBlock = ceilDiv(numOfBin, blockSize);

    // std::cout << "starting assign reads..." << std::endl;
    // calculate overlap
    gpu_assign_read_kernel<<<nBinBlock, blockSize>>>(
        d_bins, numOfBin, d_reads, numOfRead,
        d_nj_reads, numOf_nj_Read, d_readCount,
        d_nj_readCount);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // allocate and copy gaps to device global memory
    uint32_t numOfGap = h_gaps.start_.size();
    d_Gaps d_unsorted_gaps = gpu_MallocGap(h_gaps, numOfGap);
    d_Gaps d_gap_singles = gpu_MallocGap(h_gaps, numOfGap, false);
    d_Gaps d_gap_pairs = gpu_MallocGap(h_gaps, numOfGap, false);

    // sort gaps
    cubRadixSortKey(d_unsorted_gaps.start_, d_gap_singles.start_, numOfGap);
    cubRadixSortKey(d_unsorted_gaps.end_, d_gap_singles.end_, numOfGap);
    cubRadixSortInterval(d_unsorted_gaps, d_gap_pairs, numOfGap);

    // free device memory
    CUDA_SAFE_CALL(cudaFree(d_unsorted_gaps.start_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_gaps.end_));
    
    // remove gaps
    gpu_assign_read_kernel2<<<nBinBlock, blockSize>>>(
        d_bins, numOfBin, d_gap_singles, d_gap_pairs,
        numOfGap, d_readCount, d_nj_readCount,
        d_all_readCount, (mode == 2));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // copy read counts to cpu
    uint32_t readCount[numOfBin];
    uint32_t nj_readCount[numOfBin];
    uint32_t all_readCount[numOfBin]; 
    CUDA_SAFE_CALL(cudaMemcpy(readCount, d_readCount, bytes_i,
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(nj_readCount, d_nj_readCount, bytes_i,
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(all_readCount, d_all_readCount, bytes_i,
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // free device memory
    CUDA_SAFE_CALL(cudaFree(d_readCount));
    CUDA_SAFE_CALL(cudaFree(d_nj_readCount));

    // std::cout << "starting count tpm..." << std::endl;
    float *d_tempTPM, *d_tpmCount;
    // d_tempTPM
    size_t bytes_f = numOfBin * sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_tempTPM, bytes_f));
    CUDA_SAFE_CALL(cudaMemset(d_tempTPM, 0, bytes_f));
    // d_tpmCount  
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_tpmCount,  sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(d_tpmCount, 0, sizeof(float)));

    float tpmCount = 0;
    float *d_tpmStore;
    // tpmStore
    float tpmStore[numOfBin];
    // d_tpmStore
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_tpmStore, bytes_f));

    // count temporary tpms
    gpu_count_tempTPM<<<nBinBlock, blockSize>>>(d_bins, numOfBin, 
                                                d_tempTPM);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // calculate sum of tpms
    cubReduceSum(d_tempTPM, d_tpmCount, numOfBin);
    // count tpm
    gpu_count_TPM<<<nBinBlock, blockSize>>>(d_bins, numOfBin, d_tempTPM,
                                            d_tpmCount, d_tpmStore);
    CUDA_SAFE_CALL(cudaMemcpy(tpmStore, d_tpmStore, bytes_f,
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&tpmCount, d_tpmCount, sizeof(float),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // free device memory
    CUDA_SAFE_CALL(cudaFree(d_tempTPM));
    CUDA_SAFE_CALL(cudaFree(d_tpmCount));
    CUDA_SAFE_CALL(cudaFree(d_tpmStore));
    CUDA_SAFE_CALL(cudaFree(d_bins.start_));
    CUDA_SAFE_CALL(cudaFree(d_bins.end_));
    CUDA_SAFE_CALL(cudaFree(d_bins.strand));
    CUDA_SAFE_CALL(cudaFree(d_bins.core));

    // output results to file
    output_tpm(h_bins, tpmStore, tpmCount, readCount, nj_readCount,
               all_readCount, output_dir, file_prefix);
}

void Run_psi(h_ASEs &h_ases, h_Bins &h_bins,
             d_Reads d_nj_reads, d_Junctions d_junctions,
             uint32_t numOfASE, uint32_t numOfBin,
             uint32_t numOf_nj_Read, uint32_t numOfJunction,
             uint32_t ase_type, std::string &output_dir,
             std::string &file_prefix)
{
    // allocate and copy ases to device global memory
    // std::cout << "allocate gpu memory for ases..." << std::endl;
    d_ASEs d_unsorted_ases = gpu_MallocASE(h_ases, numOfASE);
    d_ASEs d_ases = gpu_MallocASE(h_ases, numOfASE, false);

    // std::cout << "starting sorting ases..." << std::endl;
    // sort ases
    cubRadixSortASE(d_unsorted_ases, d_ases, numOfASE);

    // free memory
    CUDA_SAFE_CALL(cudaFree(d_unsorted_ases.start_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_ases.end_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_ases.strand));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_ases.core));

    // allocate and copy bins to device global memory
    // std::cout << "allocate gpu memory for bins..." << std::endl;
    d_Bins d_unsorted_bins = gpu_MallocBin(h_bins, numOfBin);
    d_Bins d_bins = gpu_MallocBin(h_bins, numOfBin, false);

    // std::cout << "starting sorting bins..." << std::endl;
    // sort bins
    cubRadixSortBin(d_unsorted_bins, d_bins, h_bins, numOfBin);

    // free memory
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.start_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.end_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.strand));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.core));

    // compute number of thread block
    unsigned nBinBlock = ceilDiv(numOfBin, blockSize);
    unsigned nASEBlock = ceilDiv(numOfASE, blockSize);

    // auxiliary array for ases
    Assist d_assist_ases;
    size_t bytes_i = sizeof(uint32_t) * numOfBin;
    CUDA_SAFE_CALL(cudaMalloc((void **)&(d_assist_ases.start_), bytes_i));
    CUDA_SAFE_CALL(cudaMemset(d_assist_ases.start_, 0, bytes_i));
    CUDA_SAFE_CALL(cudaMalloc((void **)&(d_assist_ases.end_), bytes_i));
    CUDA_SAFE_CALL(cudaMemset(d_assist_ases.end_, 0, bytes_i));

    // assign ases to bins
    // std::cout << "starting assign ases..." << std::endl;
    gpu_assign_ASE_kernel<<<nBinBlock, blockSize>>>(
        d_bins, numOfBin, d_ases, numOfASE, d_assist_ases);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    ASEPsi *h_ase_psi, *d_ase_psi;
    h_ase_psi = new ASEPsi[numOfASE];
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_ase_psi, sizeof(ASEPsi) * numOfASE));

    // assign reads to ases
    // std::cout << "starting assign reads to ases..." << std::endl;
    if (ase_type == RI_Type) {
        gpu_assign_read_RI_kernel<<<nASEBlock, blockSize>>>(
            d_ases, numOfASE, d_nj_reads, numOf_nj_Read,
            d_junctions, numOfJunction, d_ase_psi);
    } else {
        gpu_assign_read_ASE_kernel<<<nASEBlock, blockSize>>>(
            d_ases, numOfASE, d_junctions, numOfJunction, d_ase_psi);
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // count psi and confidence interval
    // std::cout << "starting count psi and confidence interval..." << std::endl;
    gpu_count_PSI<<<nASEBlock*4, blockSize/4>>>(d_ases, numOfASE, d_ase_psi);
    CUDA_SAFE_CALL(cudaMemcpy(h_ase_psi, d_ase_psi, sizeof(ASEPsi) * numOfASE, 
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());


    // free device memory
    CUDA_SAFE_CALL(cudaFree(d_ase_psi));
    CUDA_SAFE_CALL(cudaFree(d_ases.start_));
    CUDA_SAFE_CALL(cudaFree(d_ases.end_));
    CUDA_SAFE_CALL(cudaFree(d_ases.strand));
    CUDA_SAFE_CALL(cudaFree(d_ases.core));
    CUDA_SAFE_CALL(cudaFree(d_assist_ases.start_));
    CUDA_SAFE_CALL(cudaFree(d_assist_ases.end_));

    // output results to file
    output_psi(h_ase_psi, numOfASE, ase_type, output_dir, file_prefix);
    
    // free host memory
    delete []h_ase_psi;
}

void Run_psi_multi_gpus(h_ASEs &h_ases, h_Bins &h_bins,
                        d_Reads d_nj_reads, d_Junctions d_junctions,
                        uint32_t numOfASE, uint32_t numOfBin,
                        uint32_t numOf_nj_Read, uint32_t numOfJunction,
                        uint32_t ase_type, std::string &output_dir,
                        std::string &file_prefix, std::vector<int>& devices)
{
    // allocate and copy ases to device global memory
    // std::cout << "allocate gpu memory for ases..." << std::endl;
    d_ASEs d_unsorted_ases = gpu_MallocASE(h_ases, numOfASE);
    d_ASEs d_ases = gpu_MallocASE(h_ases, numOfASE, false);

    // std::cout << "starting sorting ases..." << std::endl;
    // sort ases
    cubRadixSortASE(d_unsorted_ases, d_ases, numOfASE);

    // free memory
    CUDA_SAFE_CALL(cudaFree(d_unsorted_ases.start_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_ases.end_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_ases.strand));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_ases.core));

    // allocate and copy bins to device global memory
    // std::cout << "allocate gpu memory for bins..." << std::endl;
    d_Bins d_unsorted_bins = gpu_MallocBin(h_bins, numOfBin);
    d_Bins d_bins = gpu_MallocBin(h_bins, numOfBin, false);

    // std::cout << "starting sorting bins..." << std::endl;
    // sort bins
    cubRadixSortBin(d_unsorted_bins, d_bins, h_bins, numOfBin);

    // free memory
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.start_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.end_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.strand));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_bins.core));

    // compute number of thread block
    unsigned nBinBlock = ceilDiv(numOfBin, blockSize);
    unsigned nASEBlock = ceilDiv(numOfASE, blockSize);

    // auxiliary array for ases
    Assist d_assist_ases;
    size_t bytes_i = sizeof(uint32_t) * numOfBin;
    CUDA_SAFE_CALL(cudaMalloc((void **)&(d_assist_ases.start_), bytes_i));
    CUDA_SAFE_CALL(cudaMemset(d_assist_ases.start_, 0, bytes_i));
    CUDA_SAFE_CALL(cudaMalloc((void **)&(d_assist_ases.end_), bytes_i));
    CUDA_SAFE_CALL(cudaMemset(d_assist_ases.end_, 0, bytes_i));

    // assign ases to bins
    // std::cout << "starting assign ases..." << std::endl;
    gpu_assign_ASE_kernel<<<nBinBlock, blockSize>>>(
        d_bins, numOfBin, d_ases, numOfASE, d_assist_ases);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    int curr_device_id;
    CUDA_SAFE_CALL(cudaGetDevice(&curr_device_id));

    // std::cout << "starting assign reads to ases..." << std::endl;
    size_t numOfGpus = devices.size();
    unsigned nASEBlockPerGpu = ceilDiv(nASEBlock, numOfGpus);

    ASEPsi *h_ase_psi;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_ase_psi, sizeof(ASEPsi) * numOfASE));

    typedef struct {
        // ase
        d_ASEs d_ases;
        // ase psi
        ASEPsi *d_ase_psi;
        // offset
        uint32_t aseOff;
        // size
        uint32_t numOfASE;
        // nj_reads
        d_Reads d_nj_reads;
        // junctions
        d_Junctions d_junctions;
        // cuda stream
        cudaStream_t stream;
    } Tplan;

    Tplan plan[numOfGpus];

#define MIN(a, b) ((a) < (b) ? (a) : (b))

    for (uint32_t i = 0; i < numOfGpus; i++) {
        uint32_t id = devices[i];
        uint32_t blockOff = id * nASEBlockPerGpu;
        uint32_t nASEBlock_ = MIN(nASEBlockPerGpu, nASEBlock - blockOff);
        uint32_t aseOff = blockOff * blockSize;
        uint32_t numOfASE_ = MIN(nASEBlock_ * blockSize, numOfASE - aseOff);
        
        // id = devices.size() - 1 - id;
        plan[id].aseOff = aseOff;
        plan[id].numOfASE = numOfASE_;

        CUDA_SAFE_CALL(cudaSetDevice(id));
        CUDA_SAFE_CALL(cudaStreamCreate(&plan[id].stream));
        
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan[id].d_ase_psi,
                                   sizeof(ASEPsi) * numOfASE_));
        
        // d_junctions
        if (id != curr_device_id) {
            CUDA_SAFE_CALL(cudaMalloc((void **)&(plan[id].d_ases.start_),
                                       sizeof(uint64_t) * numOfASE_));
            CUDA_SAFE_CALL(cudaMalloc((void **)&(plan[id].d_ases.end_),
                                       sizeof(uint64_t) * numOfASE_));
            CUDA_SAFE_CALL(cudaMalloc((void **)&(plan[id].d_ases.strand),
                                       sizeof(uint8_t) * numOfASE_));
            CUDA_SAFE_CALL(cudaMalloc((void **)&(plan[id].d_ases.core),
                                       sizeof(ase_core_t) * numOfASE_));
            CUDA_SAFE_CALL(cudaMalloc((void **)&(plan[id].d_nj_reads.start_),
                                       sizeof(uint64_t) * numOf_nj_Read));
            CUDA_SAFE_CALL(cudaMalloc((void **)&(plan[id].d_nj_reads.end_),
                                       sizeof(uint64_t) * numOf_nj_Read));
            CUDA_SAFE_CALL(cudaMalloc((void **)&(plan[id].d_junctions.start_),
                                       sizeof(uint64_t) * numOfJunction));
            CUDA_SAFE_CALL(cudaMalloc((void **)&(plan[id].d_junctions.end_),
                                       sizeof(uint64_t) * numOfJunction));
            CUDA_SAFE_CALL(cudaMalloc((void **)&(plan[id].d_junctions.count),
                                       sizeof(uint32_t) * numOfJunction));
        } else {
            plan[id].d_ases.start_ = d_ases.start_ + plan[id].aseOff;
            plan[id].d_ases.end_ = d_ases.end_ + plan[id].aseOff;
            plan[id].d_ases.strand = d_ases.strand + plan[id].aseOff;
            plan[id].d_ases.core = d_ases.core + plan[id].aseOff;
            plan[id].d_nj_reads = d_nj_reads;
            plan[id].d_junctions = d_junctions;

        }
    }

    for (uint32_t i = 0; i < numOfGpus; i++) {
        uint32_t id = devices[i];
        // d_ases
        if (id != curr_device_id) {
            CUDA_SAFE_CALL(cudaMemcpy(plan[id].d_ases.start_,
                                      d_ases.start_ + plan[id].aseOff,
                                      sizeof(uint64_t) * plan[id].numOfASE,
                                      cudaMemcpyDefault));
            CUDA_SAFE_CALL(cudaMemcpy(plan[id].d_ases.end_,
                                      d_ases.end_ + plan[id].aseOff,
                                      sizeof(uint64_t) * plan[id].numOfASE,
                                      cudaMemcpyDefault));
            CUDA_SAFE_CALL(cudaMemcpy(plan[id].d_ases.strand,
                                      d_ases.strand + plan[id].aseOff,
                                      sizeof(uint8_t) * plan[id].numOfASE,
                                      cudaMemcpyDefault));
            CUDA_SAFE_CALL(cudaMemcpy(plan[id].d_ases.core,
                                      d_ases.core + plan[id].aseOff,
                                      sizeof(ase_core_t) * plan[id].numOfASE,
                                      cudaMemcpyDefault));
            CUDA_SAFE_CALL(cudaMemcpy(plan[id].d_nj_reads.start_,
                                      d_nj_reads.start_,
                                      sizeof(uint64_t) * numOf_nj_Read,
                                      cudaMemcpyDefault));
            CUDA_SAFE_CALL(cudaMemcpy(plan[id].d_nj_reads.end_,          
                                      d_nj_reads.end_,
                                      sizeof(uint64_t) * numOf_nj_Read,
                                      cudaMemcpyDefault));
            CUDA_SAFE_CALL(cudaMemcpy(plan[id].d_junctions.start_,
                                      d_junctions.start_,
                                      sizeof(uint64_t) * numOfJunction,
                                      cudaMemcpyDefault));
            CUDA_SAFE_CALL(cudaMemcpy(plan[id].d_junctions.end_,          
                                      d_junctions.end_,
                                      sizeof(uint64_t) * numOfJunction,
                                      cudaMemcpyDefault));
            CUDA_SAFE_CALL(cudaMemcpy(plan[id].d_junctions.count,
                                      d_junctions.count,
                                      sizeof(uint32_t) * numOfJunction,
                                      cudaMemcpyDefault));
        }
    }

    for (uint32_t i = 0; i < numOfGpus; i++) {
        uint32_t id = devices[i];
        CUDA_SAFE_CALL(cudaSetDevice(id));

        // assign reads to ases
        if (ase_type == RI_Type) {
            gpu_assign_read_RI_kernel<<<nASEBlockPerGpu, blockSize,
                                        0, plan[id].stream>>>(
                plan[id].d_ases, plan[id].numOfASE, plan[id].d_nj_reads,
                numOf_nj_Read, plan[id].d_junctions, numOfJunction,
                plan[id].d_ase_psi);
        } else {
            gpu_assign_read_ASE_kernel<<<nASEBlockPerGpu, blockSize,
                                         0, plan[id].stream>>>(
                plan[id].d_ases, plan[id].numOfASE, plan[id].d_junctions,
                numOfJunction, plan[id].d_ase_psi);
        }
        
        // count psi and confidence interval
        gpu_count_PSI<<<nASEBlockPerGpu*4, blockSize/4,
                        0, plan[id].stream>>>(
            plan[id].d_ases, plan[id].numOfASE, plan[id].d_ase_psi);
        // // copy memory to host
        CUDA_SAFE_CALL(cudaMemcpyAsync(h_ase_psi + plan[id].aseOff, plan[id].d_ase_psi,
                                       sizeof(ASEPsi) * plan[id].numOfASE,
                                       cudaMemcpyDeviceToHost, plan[id].stream));
    }

    for (uint32_t i = 0; i < numOfGpus; i++) {
        uint32_t id = devices[i];
        CUDA_SAFE_CALL(cudaSetDevice(id));
        CUDA_SAFE_CALL(cudaStreamSynchronize(plan[id].stream));

        // free device memory
        CUDA_SAFE_CALL(cudaFree(plan[id].d_ase_psi));
        if (id != curr_device_id) {
            CUDA_SAFE_CALL(cudaFree(plan[id].d_ases.start_));
            CUDA_SAFE_CALL(cudaFree(plan[id].d_ases.end_));
            CUDA_SAFE_CALL(cudaFree(plan[id].d_ases.strand));
            CUDA_SAFE_CALL(cudaFree(plan[id].d_ases.core));
        }
        CUDA_SAFE_CALL(cudaStreamDestroy(plan[id].stream));
    }

    // come back to the original device
    CUDA_SAFE_CALL(cudaSetDevice(curr_device_id));

    // free device memory
    // CUDA_SAFE_CALL(cudaFree(d_ase_psi));
    CUDA_SAFE_CALL(cudaFree(d_ases.start_));
    CUDA_SAFE_CALL(cudaFree(d_ases.end_));
    CUDA_SAFE_CALL(cudaFree(d_ases.strand));
    CUDA_SAFE_CALL(cudaFree(d_ases.core));
    CUDA_SAFE_CALL(cudaFree(d_assist_ases.start_));
    CUDA_SAFE_CALL(cudaFree(d_assist_ases.end_));

    // output results to file
    output_psi(h_ase_psi, numOfASE, ase_type, output_dir, file_prefix);

    // free host memory
    CUDA_SAFE_CALL(cudaFreeHost(h_ase_psi));
}

void Run_fusion(h_Reads &h_reads, h_Reads &h_nj_reads,
                h_Bins &h_bins, uint32_t numOfRead,
                uint32_t numOf_nj_Read, uint32_t numOfBin,
                int max_gap, std::string &output_dir,
                std::string &file_prefix)
{
    // read with junctions
    Detect_fusion(h_reads, h_bins, numOfRead, numOfBin, max_gap);
    // read without junctions
    Detect_fusion(h_nj_reads, h_bins, numOf_nj_Read, numOfBin, max_gap);

    // output results to file
    output_fusion(output_dir, file_prefix);
}

void Run_paean(h_Bins &h_bins, h_Reads &h_reads, h_Reads &h_nj_reads,
               h_Junctions &h_junctions, h_Gaps &h_gaps, 
               AMAP &h_ases_map, int gene_max_gap, int mode,
               std::string &output_dir, std::string &file_prefix,
               bool multi_gpu = false, std::vector<int> devices = {})
{
    if (!isdir(output_dir.c_str())) {
        std::cerr << "Could not find this directory: " << output_dir
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    uint32_t numOfBin = h_bins.start_.size();
    uint32_t numOfRead = h_reads.start_.size();
    uint32_t numOfJunction = h_junctions.start_.size();
    uint32_t numOf_nj_Read = h_nj_reads.start_.size();
    // uint32_t numOfASE_SE = h_se_ases.start_.size();
    // uint32_t numOfASE_A3SS = h_a3ss_ases.start_.size();
    // uint32_t numOfASE_A5SS = h_a5ss_ases.start_.size();

    if (mode == 2) {
        log_file << "numOfFragment\t" << numOfRead << "\n";
        log_file << "numOf_nj_Fragment\t" << numOf_nj_Read << "\n";
    } else {
        log_file << "numOfRead\t" << numOfRead << "\n";
        log_file << "numOf_nj_Read\t" << numOf_nj_Read << "\n";
    }
    log_file << "numOfGene\t" << numOfBin << "\n";
    log_file << "numOfJunction\t" << numOfJunction << "\n";
    // log_file << "numOfASE_SE\t" << numOfASE_SE << "\n";
    // log_file << "numOfASE_A3SS\t" << numOfASE_A3SS << "\n";
    // log_file << "numOfASE_A5SS\t" << numOfASE_A5SS << std::endl;

    // allocate and copy reads to device global memory
    // std::cout << "allocate gpu memory for reads..." << std::endl;
    d_Reads d_unsorted_reads = gpu_MallocRead(h_reads, numOfRead);
    d_Reads d_reads = gpu_MallocRead(h_reads, numOfRead, false);
    d_Reads d_unsorted_nj_reads = gpu_MallocRead(h_nj_reads, numOf_nj_Read);
    d_Reads d_nj_reads = gpu_MallocRead(h_nj_reads, numOf_nj_Read, false);

    // std::cout << "starting sorting reads..." << std::endl;
    // sort reads
    cubRadixSortKey(d_unsorted_reads.start_, d_reads.start_, numOfRead);
    cubRadixSortKey(d_unsorted_reads.end_, d_reads.end_, numOfRead);
    cubRadixSortKey(d_unsorted_nj_reads.start_, d_nj_reads.start_, numOf_nj_Read);
    cubRadixSortKey(d_unsorted_nj_reads.end_, d_nj_reads.end_, numOf_nj_Read);

    // free device memory
    CUDA_SAFE_CALL(cudaFree(d_unsorted_reads.start_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_reads.end_));
    // CUDA_SAFE_CALL(cudaFree(d_unsorted_reads.strand));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_nj_reads.start_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_nj_reads.end_));
    // CUDA_SAFE_CALL(cudaFree(d_unsorted_nj_reads.strand));

    // allocate and copy junctions to device global memory
    d_Junctions d_unsorted_junctions = gpu_MallocJunction(h_junctions,
                                                          numOfJunction);
    d_Junctions d_uncounted_junctions = gpu_MallocJunction(h_junctions,
                                                           numOfJunction,
                                                           false);
    
    // std::cout << "starting sorting junctions..." << std::endl;
    // sort junctions
    cubRadixSortJunction(d_unsorted_junctions, d_uncounted_junctions,
                         h_junctions, numOfJunction);
                         
    CUDA_SAFE_CALL(cudaFree(d_unsorted_junctions.start_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_junctions.end_));
    CUDA_SAFE_CALL(cudaFree(d_unsorted_junctions.count));

    // std::cout << "starting counting junctions..." << std::endl;
    // count junctions
    d_Junctions d_junctions = thrustSegmentedScanJunction(d_uncounted_junctions,
                                                          numOfJunction);

    CUDA_SAFE_CALL(cudaFree(d_uncounted_junctions.start_));
    CUDA_SAFE_CALL(cudaFree(d_uncounted_junctions.end_));
    CUDA_SAFE_CALL(cudaFree(d_uncounted_junctions.count));

// #define DEBUG
#ifdef DEBUG
    uint64_t *h_starts = new uint64_t[numOfJunction];
    uint64_t *h_ends = new uint64_t[numOfJunction];
    uint32_t *h_counts = new uint32_t[numOfJunction];
    CUDA_SAFE_CALL(cudaMemcpy(h_starts, d_junctions.start_,
                              sizeof(uint64_t) * numOfJunction,
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_ends, d_junctions.end_,
                              sizeof(uint64_t) * numOfJunction,
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_counts, d_junctions.count,
                              sizeof(uint32_t) * numOfJunction,
                              cudaMemcpyDeviceToHost));
    printf("%u\n", numOfJunction);
    for (int i = 0; i < numOfJunction; i++)
        printf("%u %u %u %u\n", uint32_t(h_starts[i] / refLength),
                                uint32_t(h_starts[i] & refMask),
                                uint32_t(h_ends[i] & refMask),
                                h_counts[i]);
    delete []h_starts;
    delete []h_ends;
    delete []h_counts;
#endif

    // std::cout << "starting computing tpm..." << std::endl;
    Run_tpm(d_reads, d_nj_reads, h_bins, h_gaps, numOfRead, numOf_nj_Read,
            numOfBin, mode, output_dir, file_prefix);
    
    // std::cout << "starting computing psi..." << std::endl;
    // ASEs
    for (auto &it : h_ases_map) {
        if (multi_gpu) {
            Run_psi_multi_gpus(it.second, h_bins, d_nj_reads, d_junctions,
                               it.second.start_.size(), numOfBin, numOf_nj_Read,
                               numOfJunction, it.first, output_dir, file_prefix,
                               devices);
        } else {
            Run_psi(it.second, h_bins, d_nj_reads, d_junctions,
                    it.second.start_.size(), numOfBin, numOf_nj_Read,
                    numOfJunction, it.first, output_dir, file_prefix);
        }
    }

    // free device memory
    CUDA_SAFE_CALL(cudaFree(d_reads.start_));
    CUDA_SAFE_CALL(cudaFree(d_reads.end_));
    // CUDA_SAFE_CALL(cudaFree(d_reads.strand));
    CUDA_SAFE_CALL(cudaFree(d_nj_reads.start_));
    CUDA_SAFE_CALL(cudaFree(d_nj_reads.end_));
    // CUDA_SAFE_CALL(cudaFree(d_nj_reads.strand));

    // std::cout << "starting computing fusion..." << std::endl;
    Run_fusion(h_reads, h_nj_reads, h_bins, numOfRead, numOf_nj_Read,
               numOfBin, gene_max_gap, output_dir, file_prefix);
}

#define DEAFULT_READ_MAX_GAP 5000000
#define DEAFULT_GENE_MAX_GAP 500000

static void print_usage(FILE *fp)
{
    const int num_threads = get_nprocs();
    const std::string pwd = getPwd();

    fprintf(fp,
        "Usage: paean [options...]\n"
        "Options:\n"
        "  -b,--gene            FILE     Gene annotation file with GFF3 format, required\n"
        "  -l,--length          FILE     Gene length file with csv format, required\n"
        "  -x,--ase-types       STRING   Alternative splicing types, required\n"
        "  -y,--ase-files       FILE     ASE files with csv format, required\n"
        "  -r,--read            FILE     Read file with BAM format, required\n"
        "  -o,--output          FILE     Which directory you want to write the results to (default: %s)\n"
        "  -t,--thread          INT      Number of additional threads to use to parse BAM file (default: %d)\n"
        "  -p,--read-max-gap    INT      Max allowed gap for two reads of a mate (default: %d)\n"
        "  -q,--gene-max-gap    INT      Max allowed gap for two genes for fusion detection (default: %d)\n"
        "  -m,--mode            INT      Single-end or Pair-end mode, represented by 1 and 2 respectively\n"
        "  -h,--help                     Print help information\n", 
    pwd.c_str(), num_threads, DEAFULT_READ_MAX_GAP, DEAFULT_GENE_MAX_GAP);
    exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
    const char* const short_opts = "b:l:x:y:r:o:t:p:q:m:h";
    const option long_opts[] = {
        {"bin", required_argument, nullptr, 'b'},
        {"length", required_argument, nullptr, 'l'},
        {"ase-types", required_argument, nullptr, 'x'},
        {"ase-files", required_argument, nullptr, 'y'},
        {"read", required_argument, nullptr, 'r'},
        {"output", required_argument, nullptr, 'o'},
        {"thread", required_argument, nullptr, 't'},
        {"read-max-gap", required_argument, nullptr, 'p'},
        {"gene-max-gap", required_argument, nullptr, 'q'},
        {"mode", no_argument, nullptr, 'm'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, no_argument, nullptr, 0}
    };

    struct GlobalArgs {
        char *bin_file;
        char *length_file;
        char *ase_types;
        char *ase_files;
        char *read_file;
        char *output_dir;
        int num_threads;
        int read_max_gap;
        int gene_max_gap;
        int mode;

        GlobalArgs(): bin_file(nullptr), length_file(nullptr), 
                      ase_types(nullptr), ase_files(nullptr),
                      read_file(nullptr), output_dir(nullptr),
                      num_threads(get_nprocs()), 
                      read_max_gap(DEAFULT_READ_MAX_GAP),
                      gene_max_gap(DEAFULT_GENE_MAX_GAP),
                      mode(0) {}
    };
    GlobalArgs ga;

    int c;
    while ((c = getopt_long(argc, argv, short_opts, long_opts, nullptr)) >= 0) {
        switch (c) {
        case 'b': ga.bin_file = (char *)optarg; break;
        case 'l': ga.length_file = (char *)optarg; break;
        case 'x': ga.ase_types = (char *)optarg; break;
        case 'y': ga.ase_files = (char *)optarg; break;
        case 'r': ga.read_file = (char *)optarg; break;
        case 'o': ga.output_dir = (char *)optarg; break;
        case 't': ga.num_threads = std::stoi(optarg); break;
        case 'p': ga.read_max_gap = std::stoi(optarg); break;
        case 'q': ga.gene_max_gap = std::stoi(optarg); break;
        case 'm': ga.mode = std::stoi(optarg); break;
        case 'h': print_usage(stdout); break;
        default:  print_usage(stdout); break;
        }
    }

    // check input files
    if (!ga.bin_file || !ga.length_file || !ga.read_file) {
        fprintf(stderr, "[paean] require gene annotation file, length file, read file!\n");
        exit(EXIT_FAILURE);
    }

    // check ase
    if (!ga.ase_types || (ga.ase_types && !ga.ase_files)) {
        fprintf(stderr, "[paean] require the types and files of ASE to be specified!\n");
        exit(EXIT_FAILURE);
    }

    auto ase_types = split(std::string(ga.ase_types), ',');
    auto ase_files = split(std::string(ga.ase_files), ',');
    
    // check mathced size of ASE types and files
    if (ase_types.size() != ase_files.size()) {
        fprintf(stderr, "unmatched size of the types and files of ASE!\n");
        exit(EXIT_FAILURE);
    }

    // check ASE type name
    for (int i = 0; i < ase_types.size(); i++) {
        int idx = indexOf(aseTypes, numASETypes, ase_types[i]);
        if (idx == -1) {
            fprintf(stderr, "wrong ASE type!\n");
            exit(EXIT_FAILURE);
        }
    }

    // check modes
    if (ga.mode != 1 && ga.mode != 2) {
        fprintf(stderr, "please specify one mode, only allow 1(single-end) or 2(pair-end)!\n");
        exit(EXIT_FAILURE);
    }

    std::string output_dir;
    if (ga.output_dir) {
        output_dir = std::string(ga.output_dir);
        // if not exist, create
        createDirIfNotExists(output_dir.c_str());
    } else {
        // if no output_dir, use current dir
        output_dir = getPwd();
    }

    // extract file prefix
    std::string file_prefix = extractFilePrefix(ga.read_file);

    // open log file
    std::string log_name = pathJoin(output_dir, file_prefix+"_log.txt");
    log_file.open(log_name);

    // set multi-GPU
    bool multi_gpu = false;
    std::vector<int> devices;

    if (const char* gpus = std::getenv("GPUS")) {
        auto sgpus = split(std::string(gpus), ',');
        std::transform(sgpus.begin(), sgpus.end(),
                       std::back_inserter(devices),
                       [](const std::string& str)
                       { return std::stoi(str); });
        
        // check gpus
        int count;
        CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
        if (devices[devices.size() - 1] >= count) {
            fprintf(stderr, "error device ids!\n");
            exit(EXIT_FAILURE);
        }

        if (devices.size() > 1) {
            multi_gpu = true;

            // enable p2p directly access
            uint32_t numOfGpus = devices.size();
            for (uint32_t i = 0; i < numOfGpus; i++) {
                uint32_t idi = devices[i];
                CUDA_SAFE_CALL(cudaSetDevice(idi));
                for (uint32_t j = 0; j < numOfGpus; j++) {
                    if (i == j)
                        continue;

                    uint32_t idj = devices[j];

                    int is_able;
                    CUDA_SAFE_CALL(cudaDeviceCanAccessPeer(&is_able, idi, idj));
                    if (is_able) {
                        CUDA_SAFE_CALL(cudaDeviceEnablePeerAccess(idj, 0));
                    }
                }
            }
        }
    }

    // bins and ases
    h_Bins h_bins;
    // h_ASEs h_se_ases, h_a3ss_ases, h_a5ss_ases;
    // reads
    h_Reads h_reads, h_nj_reads;
    h_Junctions h_junctions;
    h_Gaps h_gaps;

    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();

    // load reads
    // std::cout << "loading reads..." << std::endl;
    LoadReadFromBam(h_reads, h_nj_reads,
                    h_junctions, h_gaps, ga.read_file,
                    ga.num_threads, ga.read_max_gap,
                    ga.mode);
    
    // load bins
    // std::cout << "loading bins..." << std::endl;
    LoadBinFromGff(h_bins, ga.bin_file, ga.length_file);
    
    // load ases
    AMAP h_ases_map;
    for (int i = 0; i < ase_types.size(); i++) {
        size_t idx = indexOf(aseTypes, numASETypes, ase_types[i]);
        h_ASEs h_ases;
        LoadAseFromCsv(h_ases, &ase_files[i][0]);
        h_ases_map.emplace(idx, h_ases);
    }

    auto t1 = high_resolution_clock::now();


    // handling paean
    // std::cout << "start kernel program..." << std::endl;
    
    if (multi_gpu) {
        Run_paean(h_bins, h_reads, h_nj_reads,
                  h_junctions, h_gaps, h_ases_map,
                  ga.gene_max_gap, ga.mode, output_dir,
                  file_prefix, true, devices);
    } else {
        if (!devices.empty()) {
            CUDA_SAFE_CALL(cudaSetDevice(devices[0]));
        }
        Run_paean(h_bins, h_reads, h_nj_reads,
                  h_junctions, h_gaps, h_ases_map,
                  ga.gene_max_gap, ga.mode,
                  output_dir, file_prefix);
    }

    auto t2 = high_resolution_clock::now();

    // // set precision
    // std::cout.setf(std::ios::fixed);
    // std::cout.precision(1);

    // std::cout << "load spend: "
    //           << duration_cast<duration<double>>(t1-t0).count() 
    //           << " seconds" << std::endl;

    // std::cout << "computation spend: "
    //           << duration_cast<duration<double>>(t2-t1).count() 
    //           << " seconds" << std::endl;
    
    // set precision
    log_file.setf(std::ios::fixed);
    log_file.precision(1);

    log_file << "load time\t"
             << duration_cast<duration<double>>(t1-t0).count()
             << "s\n";
    log_file << "computation time\t"
             << duration_cast<duration<double>>(t2-t1).count()
             << "s\n";
    log_file.close();
}
