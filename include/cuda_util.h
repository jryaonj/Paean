#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include <cstdio>
#include <cstdlib>             /* EXIT_FAILURE */

#define CUDA_SAFE_CALL(call) {                                            \
	cudaError_t err = call;                                                    \
	if(cudaSuccess != err) {                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
				 __FILE__, __LINE__, cudaGetErrorString( err) );             \
		exit(EXIT_FAILURE);                                                  \
	} }

#define CUDA_SAFE_CALL_SYNC(call) {                                       \
	CUDA_SAFE_CALL_NO_SYNC(call);                                            \
	cudaError_t err |= cudaDeviceSynchronize();                                \
	if(cudaSuccess != err) {                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
				__FILE__, __LINE__, cudaGetErrorString( err) );              \
		exit(EXIT_FAILURE);                                                  \
	} }

#define CUDA_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } }

#endif
