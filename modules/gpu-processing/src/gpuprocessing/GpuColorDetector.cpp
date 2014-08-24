#include "cir/gpuprocessing/GpuColorDetector.h"
#include "cir/gpuprocessing/detect_color.cuh"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuColorDetector::GpuColorDetector() {

}

GpuColorDetector::~GpuColorDetector() {

}

MatWrapper GpuColorDetector::doDetectColor(MatWrapper& input, const int hueNumber,
		const int* minHues,	const int* maxHues, const int minSat, const int maxSat,
		const int minValue,	const int maxValue) {
	cv::gpu::GpuMat output(input.getGpuMat());

	detect_color(input.getGpuMat().ptr<uchar>(), hueNumber, minHues, maxHues, minSat, maxSat,
			minValue, maxValue, output.cols, output.rows, output.step, output.ptr<uchar>());
	return output;
}

}}
