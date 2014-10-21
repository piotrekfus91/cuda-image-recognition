#include "cir/gpuprocessing/GpuColorDetector.h"
#include "cir/gpuprocessing/detect_color.cuh"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuColorDetector::GpuColorDetector() {

}

GpuColorDetector::~GpuColorDetector() {

}

MatWrapper GpuColorDetector::doDetectColor(const MatWrapper& input, const int hsvRangesNumber,
		const OpenCvHsvRange* hsvRanges) {
	cv::gpu::GpuMat output = input.clone().getGpuMat();

	detect_color(output.data, hsvRangesNumber, hsvRanges, output.cols, output.rows, output.step, output.data);

	MatWrapper outputMw(output);
	outputMw.setColorScheme(input.getColorScheme());
	return outputMw;
}

}}
