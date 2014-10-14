#include <iostream>
#include "cir/common/cuda_host_util.cuh"
#include "cir/gpuprocessing/count_moments.cuh"

#define THREADS_IN_BLOCK 4
#define THREADS_PER_BLOCK THREADS_IN_BLOCK * THREADS_IN_BLOCK

namespace cir { namespace gpuprocessing {

double count_raw_moment(uchar* data, int width, int height, int step, int p, int q) {
	int horizontalBlocks = (width + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
	int verticalBlocks = (height + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
	int totalBlocks = horizontalBlocks * verticalBlocks;

	double* blockSums = (double*) malloc(sizeof(double) * totalBlocks);
	double* d_blockSums;

	HANDLE_CUDA_ERROR(cudaMalloc((void**) &d_blockSums, sizeof(double) * totalBlocks));

	HANDLE_CUDA_ERROR(cudaMemcpy(d_blockSums, blockSums, sizeof(double) * totalBlocks, cudaMemcpyHostToDevice));

	uchar* d = (uchar*) malloc(sizeof(uchar) * step * height);
	HANDLE_CUDA_ERROR(cudaMemcpy(d, data, sizeof(uchar) * step * height, cudaMemcpyDeviceToHost));

	// TODO kernel dims
	dim3 blocks(horizontalBlocks, verticalBlocks);
	dim3 threads(THREADS_IN_BLOCK, THREADS_IN_BLOCK);
	k_count_raw_moment<<<blocks, threads>>>(data, width, height, step, p, q, d_blockSums);
	HANDLE_CUDA_ERROR(cudaGetLastError());

	HANDLE_CUDA_ERROR(cudaMemcpy(blockSums, d_blockSums, sizeof(double) * totalBlocks, cudaMemcpyDeviceToHost));

	double ret = 0;
	for(int i = 0; i < totalBlocks; i++) {
		ret += blockSums[i];
	}

	HANDLE_CUDA_ERROR(cudaFree(d_blockSums));
	free(blockSums);

	return ret;
}

__global__
void k_count_raw_moment(uchar* data, int width, int height, int step, int p, int q, double* blockSums) {
	__shared__ double cache[THREADS_PER_BLOCK];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int idx = x + y * step;
	int pixel = data[idx];

	int cacheIdx = threadIdx.x + blockDim.x * threadIdx.y;

	if(p == 0 && q == 0)
		cache[cacheIdx] = pixel;
	else if(p == 0)
		cache[cacheIdx] = pow(1.0 * y, 1.0 * q) * pixel;
	else if(q == 0)
		cache[cacheIdx] = pow(1.0 * x, 1.0 * p) * pixel;
	else
		cache[cacheIdx] = pow(1.0 * x, 1.0 * p) * pow(1.0 * y, 1.0 * q) * pixel;

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

}}
