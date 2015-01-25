#include "cir/gpuprocessing/GpuBlurer.h"
#include "cir/gpuprocessing/gpu_blur.cuh"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuBlurer::GpuBlurer() {

}

GpuBlurer::~GpuBlurer() {

}

MatWrapper GpuBlurer::doMedian(const MatWrapper& mw, int size) {
	cv::gpu::GpuMat orig = mw.getGpuMat();
	cv::gpu::GpuMat outputMat = orig.clone();

	median_blur(orig.data, outputMat.data, orig.cols, orig.rows, size, orig.step);

	MatWrapper outputMw(outputMat);
	outputMw.setColorScheme(mw.getColorScheme());
	return outputMw;
}

}}
