#include "cir/gpuprocessing/GpuBlurer.h"
#include "cir/gpuprocessing/gpu_blur.cuh"
#include "cir/common/concurrency/StreamHandler.h"

using namespace cir::common;
using namespace cir::common::concurrency;

namespace cir { namespace gpuprocessing {

GpuBlurer::GpuBlurer() {

}

GpuBlurer::~GpuBlurer() {

}

MatWrapper GpuBlurer::doMedian(const MatWrapper& mw, int size) {
	cv::gpu::GpuMat orig = mw.getGpuMat();
	cv::gpu::GpuMat outputMat = orig.clone();

	median_blur(orig.data, outputMat.data, orig.cols, orig.rows, size, orig.step,
			StreamHandler::nativeStream());

	MatWrapper outputMw(outputMat);
	outputMw.setColorScheme(mw.getColorScheme());
	return outputMw;
}

}}
