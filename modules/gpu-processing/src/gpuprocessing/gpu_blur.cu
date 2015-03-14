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

//	KERNEL_MEASURE_START

	if(size == 1)
		k_median_blur<1><<<blocks, threads, 0, stream>>>(origData, cloneData, width, height, step);
	else if(size == 2)
		k_median_blur<2><<<blocks, threads, 0, stream>>>(origData, cloneData, width, height, step);
	else if(size == 3)
		k_median_blur<3><<<blocks, threads, 0, stream>>>(origData, cloneData, width, height, step);
	HANDLE_CUDA_ERROR(cudaGetLastError());

//	KERNEL_MEASURE_END("Median")
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
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	if(x >= width)
		return;

	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if(y >= height)
		return;

	uchar surround[(2*SIZE+1) * (2*SIZE+1)];
	int total = 0;

	for(int i = x - SIZE; i <= x + SIZE; i++) {
		for(int j = y - SIZE; j <= y + SIZE; j++) {
			if(i >= 0 && i < width && j >= 0 && j < height) {
				surround[total++] = origData[j * step + i];
			}
		}
	}
	sort<uchar>(surround, total);
	uchar u = total % 2 == 0 ? (surround[total / 2] + surround[total / 2 - 1]) / 2 : surround[total / 2 - 1];
	cloneData[(x + step * y)] = u;
}

}}
