#include <iostream>
#include "cir/common/cuda_host_util.cuh"
#include "cir/gpuprocessing/count_moments.cuh"
#include "cir/common/logger/Logger.h"

#define THREADS_IN_BLOCK 16
#define THREADS_PER_BLOCK THREADS_IN_BLOCK * THREADS_IN_BLOCK

using namespace cir::common;
using namespace cir::common::logger;

namespace cir { namespace gpuprocessing {

int horizontalBlocks;
int verticalBlocks;
int totalBlocks;

double* zeroBlockSums;
double* blockSums;
double* d_blockSums;

void count_raw_moment_init(int width, int height) {
	horizontalBlocks = (width + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
	verticalBlocks = (height + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
	totalBlocks = horizontalBlocks * verticalBlocks;

	zeroBlockSums = (double*) malloc(sizeof(double) * totalBlocks);
	for(int i = 0; i < totalBlocks; i++) {
		zeroBlockSums[i] = 0;
	}
	blockSums = (double*) malloc(sizeof(double) * totalBlocks);

	HANDLE_CUDA_ERROR(cudaMalloc((void**) &d_blockSums, sizeof(double) * totalBlocks));
}

double count_raw_moment(uchar* data, int width, int height, int step, int p, int q) {
	HANDLE_CUDA_ERROR(cudaMemcpy(d_blockSums, zeroBlockSums, sizeof(double) * totalBlocks, cudaMemcpyHostToDevice));

	// TODO kernel dims
	dim3 blocks(horizontalBlocks, verticalBlocks);
	dim3 threads(THREADS_IN_BLOCK, THREADS_IN_BLOCK);

	KERNEL_MEASURE_START

	k_count_raw_moment<<<blocks, threads>>>(data, width, height, step, p, q, d_blockSums);
	HANDLE_CUDA_ERROR(cudaGetLastError());

	KERNEL_MEASURE_END("Count Hu moments")

	HANDLE_CUDA_ERROR(cudaMemcpy(blockSums, d_blockSums, sizeof(double) * totalBlocks, cudaMemcpyDeviceToHost));

	double ret = 0;
	for(int i = 0; i < totalBlocks; i++) {
		ret += blockSums[i];
	}

	return ret;
}

void count_raw_moment_shutdown() {
	HANDLE_CUDA_ERROR(cudaFree(d_blockSums));
	free(blockSums);
}

__global__
void k_count_raw_moment(uchar* data, int width, int height, int step, int p, int q, double* blockSums) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= width)
		return;

	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(y >= height)
		return;

	__shared__ double cache[THREADS_PER_BLOCK];

	int idx = x + y * step;
	double pixel = data[idx];

	int cacheIdx = threadIdx.x + blockDim.x * threadIdx.y;

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
