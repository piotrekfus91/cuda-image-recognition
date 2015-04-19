#include "cir/cpuprocessing/CpuRedMarker.h"

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuRedMarker::CpuRedMarker() {

}

CpuRedMarker::~CpuRedMarker() {

}

MatWrapper CpuRedMarker::markSegments(MatWrapper input, const SegmentArray* segmentArray) {
	cv::Mat outputMat = input.getMat();
	MatWrapper outputMw(outputMat);

	// TODO coloring of many segments?
	for(int i = 0; i < segmentArray->size; i++) {
		Segment* segment = segmentArray->segments[i];
		mark(outputMw, segment, 255, 0, 0);
	}

	outputMw.setColorScheme(input.getColorScheme());
	return outputMw;
}

MatWrapper CpuRedMarker::markPairs(MatWrapper input, std::vector<std::pair<Segment*, int> > pairs) {
	cv::Mat outputMat = input.getMat();
	MatWrapper outputMw(outputMat);

	for(std::vector<std::pair<Segment*, int> >::iterator it = pairs.begin(); it != pairs.end(); it++) {
		std::pair<Segment*, int> pair = *it;
		Segment* segment = pair.first;
		if(pair.second % 3 == 0) {
			mark(outputMw, segment, 255, 0, 0);
		} else if(pair.second % 3 == 1) {
			mark(outputMw, segment, 0, 255, 0);
		} else {
			mark(outputMw, segment, 0, 0, 255);
		}
	}

	outputMw.setColorScheme(input.getColorScheme());
	return outputMw;
}

void CpuRedMarker::mark(MatWrapper& mw, Segment* segment, int red, int green, int blue) {
	cv::Scalar color;
	color.val[0] = blue;
	color.val[1] = green;
	color.val[2] = red;

	cv::Point p1;
	p1.x = segment->leftX;
	p1.y = segment->bottomY;

	cv::Point p2;
	p2.x = segment->rightX;
	p2.y = segment->topY;

	cv::Mat mat = mw.getMat();
	cv::rectangle(mat, p1, p2, color, 3);
}

}}
