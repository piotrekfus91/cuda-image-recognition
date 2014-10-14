#include "cir/gpuprocessing/detect_color.cuh"
#include "cir/common/cuda_host_util.cuh"
#include <iostream>

// function is applicable only for HSV model
#define channels 3

namespace cir { namespace gpuprocessing {

void detect_color(uchar* src, const int hueNumber, const int* minHues, const int* maxHues,
		const int minSat, const int maxSat, const int minValue, const int maxValue,
		const int width, const int height, const int step, uchar* dst) {
	int size = hueNumber * sizeof(int);
	int* d_minHues;
	int* d_maxHues;

	HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_minHues, size));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_maxHues, size));

	HANDLE_CUDA_ERROR(cudaMemcpy(d_minHues, minHues, size, cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_maxHues, maxHues, size, cudaMemcpyHostToDevice));

	// TODO kernel dims
	dim3 block((width+15)/16, (height+15)/16);
	dim3 thread(16, 16);

	cudaEvent_t start;
	cudaEvent_t stop;
	HANDLE_CUDA_ERROR(cudaEventCreate(&start));
	HANDLE_CUDA_ERROR(cudaEventCreate(&stop));
	HANDLE_CUDA_ERROR(cudaEventRecord(start, 0)); // TODO stream

	k_detect_color<<<block, thread>>>(src, hueNumber, d_minHues, d_maxHues, minSat, maxSat,
			minValue, maxValue, width, height, step, dst);
	HANDLE_CUDA_ERROR(cudaGetLastError());

	HANDLE_CUDA_ERROR(cudaEventRecord(stop, 0));
	HANDLE_CUDA_ERROR(cudaEventSynchronize(stop));

	float time;
	HANDLE_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
	HANDLE_CUDA_ERROR(cudaEventDestroy(start));
	HANDLE_CUDA_ERROR(cudaEventDestroy(stop));

	std::cout << "time: " << time << std::endl;

	HANDLE_CUDA_ERROR(cudaFree(d_minHues));
	HANDLE_CUDA_ERROR(cudaFree(d_maxHues));
}

__global__
void k_detect_color(uchar* src, const int hueNumber, const int* minHues, const int* maxHues,
		const int minSat, const int maxSat, const int minValue, const int maxValue,
		const int width, const int height, const int step, uchar* dst) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x > width || y > height)
		return;

	int pos = x * channels + y * step;

	int hue = dst[pos];
	int sat = dst[pos+1];
	int value = dst[pos+2];

	bool clear = true;

	if(sat >= minSat && sat <= maxSat
			&& value >= minValue && value <= maxValue) {

		for(int i = 0; i < hueNumber; i++) {
			int minHue = minHues[i];
			int maxHue = maxHues[i];

			if(minHue <= maxHue) {
				if(hue >= minHue && hue <= maxHue) {
					clear = false;
					break;
				}
			} else {
				if(hue >= minHue || hue <= maxHue) {
					clear = false;
					break;
				}
			}
		}

	}

	if(clear) {
		dst[pos] = 0;
		dst[pos+1] = 0;
		dst[pos+2] = 0;
	}
}

}}
