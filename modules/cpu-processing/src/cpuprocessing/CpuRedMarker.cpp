#include "cir/cpuprocessing/CpuRedMarker.h"

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuRedMarker::CpuRedMarker() {

}

CpuRedMarker::~CpuRedMarker() {

}

MatWrapper CpuRedMarker::markSegments(MatWrapper& input, const SegmentArray* segmentArray) {
	cv::Mat outputMat = input.getMat().clone();

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

		cv::rectangle(outputMat, p1, p2, color, 3);
	}

	MatWrapper outputMw(outputMat);
	outputMw.setColorScheme(input.getColorScheme());
	return outputMw;
}

}}
