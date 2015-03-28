#include <iostream>
#include "cir/common/config.h"
#include "cir/common/cuda_host_util.cuh"
#include "cir/gpuprocessing/count_moments.cuh"
#include "cir/common/logger/Logger.h"

#define THREADS_IN_BLOCK 16
#define THREADS_PER_BLOCK THREADS_IN_BLOCK * THREADS_IN_BLOCK

using namespace cir::common;
using namespace cir::common::logger;

namespace cir { namespace gpuprocessing {

double count_raw_moment(uchar* data, int width, int height, int step, int p, int q,
		cudaStream_t stream) {
	int horizontalBlocks = (width + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
	int verticalBlocks = (height + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
	int totalBlocks = horizontalBlocks * verticalBlocks;

	double* blockSums;
	cudaHostAlloc((void**) &blockSums, sizeof(double) * totalBlocks, cudaHostAllocDefault);
	for(int i = 0; i < totalBlocks; i++) {
		blockSums[i] = 0.;
	}

	double* d_blockSums;

	HANDLE_CUDA_ERROR(cudaMalloc((void**) &d_blockSums, sizeof(double) * totalBlocks));

	HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_blockSums, blockSums, sizeof(double) * totalBlocks, cudaMemcpyHostToDevice,
			stream));

	// TODO kernel dims
	dim3 blocks(horizontalBlocks, verticalBlocks);
	dim3 threads(THREADS_IN_BLOCK, THREADS_IN_BLOCK);

	//KERNEL_MEASURE_START

	k_count_raw_moment<<<blocks, threads, 0, stream>>>(data, width, height, step, p, q, d_blockSums);
	HANDLE_CUDA_ERROR(cudaGetLastError());

	//KERNEL_MEASURE_END("Count Hu moments")

	HANDLE_CUDA_ERROR(cudaMemcpyAsync(blockSums, d_blockSums, sizeof(double) * totalBlocks, cudaMemcpyDeviceToHost,
			stream));

	HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

	double ret = 0;
	for(int i = 0; i < totalBlocks; i++) {
		ret += blockSums[i];
	}

	HANDLE_CUDA_ERROR(cudaFree(d_blockSums));
	HANDLE_CUDA_ERROR(cudaFreeHost(blockSums));

	return ret;
}

__global__
void k_count_raw_moment(uchar* data, int width, int height, int step, int p, int q, double* blockSums) {
	__shared__ double cache[THREADS_PER_BLOCK];
	int cacheIdx = threadIdx.x + blockDim.x * threadIdx.y;
	cache[cacheIdx] = 0;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= width)
		return;

	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(y >= height)
		return;

	int idx = x + y * step;
#if MOMENTS_BINARY
	double pixel = data[idx] == 0 ? 0. : 1.;
#else
	double pixel = data[idx];
#endif

	if(p == 0 && q == 0)
		cache[cacheIdx] = pixel;
	else if(p == 0)
		cache[cacheIdx] = pow(y, q) * pixel;
	else if(q == 0)
		cache[cacheIdx] = pow(x, p) * pixel;
	else
		cache[cacheIdx] = pow(x, p) * pow(y, q) * pixel;

	__syncthreads();

	for(int i = THREADS_PER_BLOCK / 2; i != 0; i /= 2) {
		if(cacheIdx < i) {
			cache[cacheIdx] += cache[cacheIdx + i];
		}
		__syncthreads();
	}

	if(cacheIdx == 0)
		blockSums[blockIdx.x + blockIdx.y * gridDim.x] = cache[0];
}

__device__
int pow(int p, int q) {
	if(q == 0)
		return 1;

	if(q == 1)
		return p;

	if(q == 2)
		return p * p;

	if(q == 3)
		return p * p * p;

	return -1; // should never happen!
}

}}
