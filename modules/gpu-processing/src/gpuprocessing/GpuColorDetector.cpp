#include "cir/gpuprocessing/GpuColorDetector.h"
#include "cir/gpuprocessing/detect_color.cuh"
#include "cir/common/concurrency/StreamHandler.h"

using namespace cir::common;
using namespace cir::common::concurrency;

namespace cir { namespace gpuprocessing {

GpuColorDetector::GpuColorDetector() {

}

GpuColorDetector::~GpuColorDetector() {

}

MatWrapper GpuColorDetector::doDetectColor(const MatWrapper& input, const int hsvRangesNumber,
		const OpenCvHsvRange* hsvRanges) {
	cv::gpu::GpuMat output = input.clone().getGpuMat();

	detect_color(output.data, hsvRangesNumber, hsvRanges, output.cols, output.rows, output.step, output.data,
			StreamHandler::nativeStream());

	MatWrapper outputMw(output);
	outputMw.setColorScheme(input.getColorScheme());
	return outputMw;
}

}}
