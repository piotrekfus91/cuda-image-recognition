#include "cir/gpuprocessing/GpuRedMarker.h"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuRedMarker::GpuRedMarker() {

}

GpuRedMarker::~GpuRedMarker() {

}

MatWrapper GpuRedMarker::markSegments(MatWrapper input, const SegmentArray* segmentArray) {
	cv::Mat mat;
	cv::gpu::GpuMat gpuMat = input.getGpuMat();
	gpuMat.download(mat);

	cv::Scalar color;
	color.val[0] = 0;
	color.val[1] = 0;
	color.val[2] = 255;

	for(int i = 0; i < segmentArray->size; i++) {
		Segment* segment = segmentArray->segments[i];

		cv::Point p1;
		p1.x = segment->leftX;
		p1.y = segment->bottomY;

		cv::Point p2;
		p2.x = segment->rightX;
		p2.y = segment->topY;

		cv::rectangle(mat, p1, p2, color, 3);
	}

	gpuMat.upload(mat);

	MatWrapper outputMw(gpuMat);
	outputMw.setColorScheme(input.getColorScheme());
	return input;
}

}}
