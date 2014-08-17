#include "cir/gpuprocessing/GpuColorDetector.h"
#include "cir/gpuprocessing/detect_color.cuh"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuColorDetector::GpuColorDetector() {

}

GpuColorDetector::~GpuColorDetector() {

}

MatWrapper GpuColorDetector::doDetectColor(MatWrapper& input, const int minHue,
		const int maxHue, const int minSat,	const int maxSat, const int minValue,
		const int maxValue) {
	cv::gpu::GpuMat output(input.getGpuMat());

	detect_color(input.getGpuMat().ptr<uchar>(), minHue, maxHue, minSat, maxSat,
			minValue, maxValue, output.cols, output.rows, output.step, output.ptr<uchar>());
	return output;
}

}}
