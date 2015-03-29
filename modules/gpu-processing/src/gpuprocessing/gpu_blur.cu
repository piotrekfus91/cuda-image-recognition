#include "cir/gpuprocessing/gpu_blur.cuh"
#include "cir/common/cuda_host_util.cuh"
#include "cir/common/logger/Logger.h"

#define THREADS 16

using namespace cir::common;
using namespace cir::common::logger;

namespace cir { namespace gpuprocessing {

void median_blur(uchar* origData, uchar* cloneData, int width, int height, int size, int step,
		cudaStream_t stream) {
	dim3 blocks((width+THREADS-1)/THREADS, (height+THREADS-1)/THREADS);
	dim3 threads(THREADS, THREADS);

//	KERNEL_MEASURE_START(stream)

	if(size == 1)
		k_median_blur<1><<<blocks, threads, 0, stream>>>(origData, cloneData, width, height, step);
	else if(size == 2)
		k_median_blur<2><<<blocks, threads, 0, stream>>>(origData, cloneData, width, height, step);
	else if(size == 3)
		k_median_blur<3><<<blocks, threads, 0, stream>>>(origData, cloneData, width, height, step);
	HANDLE_CUDA_ERROR(cudaGetLastError());

//	KERNEL_MEASURE_END("Median", stream)
}

template<class T>
__device__
void sort(T* arr, int size) {
	for(int i = 0; i < size; i++) {
		for(int j = i + 1; j < size; j++) {
			T t1 = arr[i];
			T t2 = arr[j];
			if(t2 < t1) {
				arr[i] = t2;
				arr[j] = t1;
			}
		}
	}
}

template <int SIZE>
__global__
void k_median_blur(uchar* origData, uchar* cloneData, int width, int height, int step) {
	__shared__ char cache[(THREADS+2) * (THREADS+2)];
	__shared__ int x[1];
	x[0] = threadIdx.x + blockDim.x * blockIdx.x;
	if(x[0] >= width)
		return;

	int y;
	y = threadIdx.y + blockDim.y * blockIdx.y;
	if(y >= height)
		return;

	int globalIdx = x[0] + step * y;

	int cacheIdx = threadIdx.x + 1 + (blockDim.x + 2) * (threadIdx.y + 1);
	cache[cacheIdx] = origData[globalIdx];

	if(threadIdx.x == 0)
		cache[(threadIdx.y + 1) * (blockDim.x + 2)] = blockIdx.x == 0 ? 0 : origData[globalIdx - 1];

	if(threadIdx.x == blockDim.x - 1)
		cache[(threadIdx.y + 1) * (blockDim.x + 2) + blockDim.x + 1] = blockIdx.x == gridDim.x ? 0 : origData[globalIdx + 1];

	if(threadIdx.y == 0)
		cache[threadIdx.x] = blockIdx.y == 0 ? 0 : origData[globalIdx - (blockDim.x + 2)];

	if(threadIdx.y == blockDim.y - 1)
		cache[threadIdx.x + 1 + (blockDim.x + 2) * (blockDim.y + 1)] = blockIdx.y == gridDim.y ? 0 : origData[globalIdx + (blockDim.x + 2)];

	uchar surround[(2*SIZE+1) * (2*SIZE+1)];
	int total = 0;

	for(int i = x[0] - SIZE, i2 = threadIdx.x; i <= x[0] + SIZE; i++, i2++) {
		for(int j = y - SIZE, j2 = threadIdx.y; j <= y + SIZE; j++, j2++) {
			surround[total++] = cache[j2 * (blockDim.x + 2) + i2];
		}
	}
	sort<uchar>(surround, total);
	uchar u = total % 2 == 0 ? (surround[total / 2] + surround[total / 2 - 1]) / 2 : surround[total / 2 - 1];
	cloneData[globalIdx] = u;
}

}}
