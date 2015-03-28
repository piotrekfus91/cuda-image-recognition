#include <iostream>
#include "cir/common/config.h"
#include "cir/common/cuda_host_util.cuh"
#include "cir/gpuprocessing/count_moments.cuh"
#include "cir/common/logger/Logger.h"

#define RAW_MOMENTS 10
#define IN_PASS 2
#define THREADS_IN_BLOCK 16
#define THREADS_PER_BLOCK THREADS_IN_BLOCK * THREADS_IN_BLOCK

using namespace cir::common;
using namespace cir::common::logger;

namespace cir { namespace gpuprocessing {

void count_raw_moments(uchar* data, int width, int height, int step, double* rawMoments,
		cudaStream_t stream) {
	int horizontalBlocks = (width + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
	int verticalBlocks = (height + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
	int totalBlocks = horizontalBlocks * verticalBlocks;
	int totalToAlloc = totalBlocks * IN_PASS;

	long* blockSums;
	cudaHostAlloc((void**) &blockSums, sizeof(long) * totalToAlloc, cudaHostAllocDefault);

	long* d_blockSums;
	HANDLE_CUDA_ERROR(cudaMalloc((void**) &d_blockSums, sizeof(long) * totalToAlloc));

	for(int i = 0; i < RAW_MOMENTS; i++) {
		rawMoments[i] = 0.;
	}

	// TODO kernel dims
	dim3 blocks(horizontalBlocks, verticalBlocks);
	dim3 threads(THREADS_IN_BLOCK, THREADS_IN_BLOCK);

	for(int pass = 0; pass < (RAW_MOMENTS + IN_PASS - 1) / IN_PASS; pass++) {
		for(int i = 0; i < totalToAlloc; i++) {
			blockSums[i] = 0;
		}

		HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_blockSums, blockSums, sizeof(long) * totalToAlloc, cudaMemcpyHostToDevice,
				stream));

		KERNEL_MEASURE_START(stream)

		k_count_raw_moment<<<blocks, threads, 0, stream>>>(data, width, height, step, pass, d_blockSums);
		HANDLE_CUDA_ERROR(cudaGetLastError());

		KERNEL_MEASURE_END("Count Hu moments", stream)

		HANDLE_CUDA_ERROR(cudaMemcpyAsync(blockSums, d_blockSums, sizeof(long) * totalToAlloc, cudaMemcpyDeviceToHost,
				stream));

		HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

		for(int j = 0; j < totalBlocks; j++) {
			for(int i = 0; i < IN_PASS; i++) {
				rawMoments[pass * IN_PASS + i] += blockSums[j * IN_PASS + i];
			}
		}
	}

	HANDLE_CUDA_ERROR(cudaFree(d_blockSums));
	HANDLE_CUDA_ERROR(cudaFreeHost(blockSums));
}

__global__
void k_count_raw_moment(uchar* data, int width, int height, int step, int pass, long* blockSums) {
	__shared__ long cache[THREADS_PER_BLOCK * IN_PASS];
	int tid = threadIdx.x + blockDim.x * threadIdx.y;
	int cacheIdx = tid * IN_PASS;
	for(int i = 0; i < IN_PASS; i++) {
		cache[cacheIdx + i] = 0;
	}

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= width)
		return;

	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(y >= height)
		return;

	int idx = x + y * step;
#if MOMENTS_BINARY
	int pixel = data[idx] == 0 ? 0 : 1;
#else
	int pixel = data[idx];
#endif

	if(pass == 0) {
	/* M00 */ cache[cacheIdx + 0] = pixel;
	/* M01 */ cache[cacheIdx + 1] = pixel * y;
	} else if(pass == 1) {
	/* M10 */ cache[cacheIdx + 0] = pixel * x;
	/* M11 */ cache[cacheIdx + 1] = pixel * x * y;
	} else if(pass == 2) {
	/* M02 */ cache[cacheIdx + 0] = pixel * y * y;
	/* M20 */ cache[cacheIdx + 1] = pixel * x * x;
	} else if(pass == 3) {
	/* M21 */ cache[cacheIdx + 0] = pixel * x * x * y;
	/* M12 */ cache[cacheIdx + 1] = pixel * x * y * y;
	} else if(pass == 4) {
	/* M30 */ cache[cacheIdx + 0] = pixel * x * x * x;
	/* M03 */ cache[cacheIdx + 1] = pixel * y * y * y;
	}

	__syncthreads();

	for(int j = THREADS_PER_BLOCK / 2; j != 0; j /= 2) {
		if(tid < j) {
			for(int i = 0; i < IN_PASS; i++) {
				cache[cacheIdx + i] += cache[cacheIdx + i + j * IN_PASS];
			}
		}
		__syncthreads();
	}

	if(tid == 0) {
		for(int i = 0; i < IN_PASS; i++) {
			blockSums[i + (blockIdx.x + blockIdx.y * gridDim.x) * IN_PASS] = cache[i];
		}
	}
}

}}
