#include "cir/gpuprocessing/detect_color.cuh"
#include "cir/common/cuda_host_util.cuh"
#include <iostream>

using namespace cir::common;
using namespace cir::common::logger;

// function is applicable only for HSV model
#define channels 3

namespace cir { namespace gpuprocessing {

void detect_color(uchar* src, const int hsvRangesNumber, const OpenCvHsvRange* hsvRanges,
		const int width, const int height, const int step, uchar* dst) {
	int size = hsvRangesNumber * sizeof(OpenCvHsvRange);
	OpenCvHsvRange* d_hsvRanges;

	HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_hsvRanges, size));

	HANDLE_CUDA_ERROR(cudaMemcpy(d_hsvRanges, hsvRanges, size, cudaMemcpyHostToDevice));

	// TODO kernel dims
	dim3 block((width+15)/16, (height+15)/16);
	dim3 thread(16, 16);

	KERNEL_MEASURE_START

	k_detect_color<<<block, thread>>>(src, hsvRangesNumber, d_hsvRanges, width, height, step, dst);
	HANDLE_CUDA_ERROR(cudaGetLastError());

	KERNEL_MEASURE_END("Detect color")

	HANDLE_CUDA_ERROR(cudaFree(d_hsvRanges));
}

__global__
void k_detect_color(uchar* src, const int hsvRangesNumber, const OpenCvHsvRange* hsvRanges,
		const int width, const int height, const int step, uchar* dst) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= width)
		return;

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= height)
		return;

	int pos = x * channels + y * step;

	int hue = dst[pos];
	int saturation = dst[pos+1];
	int value = dst[pos+2];

	bool clear = true;

	for(int i = 0; i < hsvRangesNumber; i++) {
		OpenCvHsvRange hsvRange = hsvRanges[i];
		OpenCvHsv less = hsvRange.less;
		OpenCvHsv greater = hsvRange.greater;

		if(saturation >= less.saturation && saturation <= greater.saturation
				&& value >= less.value && value <= greater.value) {
			if(less.hue <= greater.hue) {
				if(hue >= less.hue && hue <= greater.hue) {
					clear = false;
					break;
				}
			} else {
				if(hue >= less.hue || hue <= greater.hue) {
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
