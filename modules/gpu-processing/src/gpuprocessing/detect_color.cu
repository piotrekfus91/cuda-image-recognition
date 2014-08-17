#include "cir/gpuprocessing/detect_color.cuh"
#include <iostream>

// function is applicable only for HSV model
#define channels 3

namespace cir { namespace gpuprocessing {

void detect_color(uchar* src, const int minHue, const int maxHue, const int minSat,
		const int maxSat, const int minValue, const int maxValue, const int width,
		const int height, const int step, uchar* dst) {
	dim3 block(32, 32);
	dim3 thread(width/32, height/32);
	k_detect_color<<<block, thread>>>(src, minHue, maxHue, minSat, maxSat,
			minValue, maxValue, width, height, step, dst);
}

__global__
void k_detect_color(uchar* src, const int minHue, const int maxHue, const int minSat,
		const int maxSat, const int minValue, const int maxValue, const int width,
		const int height, const int step, uchar* dst) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x > width || y > height)
		return;

	int pos = x * channels + y * step;

	int hue = dst[pos];
	int sat = dst[pos+1];
	int value = dst[pos+2];

	if(hue < minHue || hue > maxHue
			|| sat < minSat || hue > maxSat
			|| value < minValue || value > maxValue) {
		dst[pos] = 0;
		dst[pos+1] = 0;
		dst[pos+2] = 0;
	}
}

}}
