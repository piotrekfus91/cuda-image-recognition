#include "cir/cpuprocessing/CpuRedMarker.h"

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuRedMarker::CpuRedMarker() {

}

CpuRedMarker::~CpuRedMarker() {

}

MatWrapper CpuRedMarker::markSegments(MatWrapper& input, SegmentArray* segmentArray) {
	MatWrapper output = input.clone();
	cv::Mat mat = output.getMat();

	cv::Scalar color;
	color.val[0] = 0; // hue
	color.val[1] = 255; // sat
	color.val[2] = 255; // value

	for(int i = 0; i < segmentArray->size; i++) {
		Segment* segment = segmentArray->segments[i];

		cv::Point p1;
		p1.x = segment->leftX;
		p1.y = segment->bottomY;

		cv::Point p2;
		p2.x = segment->rightX;
		p2.y = segment->topY;

		cv::rectangle(mat, p1, p2, color);
	}
	return mat;
}

}}
