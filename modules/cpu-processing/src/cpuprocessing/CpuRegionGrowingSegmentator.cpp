#include <list>
#include "cir/cpuprocessing/CpuRegionGrowingSegmentator.h"
#include "cir/common/Point.h"
#include "cir/common/SegmentArray.h"

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuRegionGrowingSegmentator::CpuRegionGrowingSegmentator() {

}

CpuRegionGrowingSegmentator::~CpuRegionGrowingSegmentator() {

}

SegmentArray* CpuRegionGrowingSegmentator::segmentate(const MatWrapper& input) {
	MatWrapper copy = input.clone();
	cv::Mat mat = copy.getMat();
	uchar* data = mat.ptr<uchar>();
	int width = mat.cols;
	int height = mat.rows;
	int step = mat.step;
	int channels = mat.channels();
	std::list<Segment*> segments;

	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			if(isApplicable(data, channels, step, x, y)) {
				Segment* segment = performNonRecursiveSegmentation(data, channels, step,
						width, height, x, y);
				if(isSegmentApplicable(segment)) {
					segments.push_back(segment);
				}
			}
		}
	}

	Segment** list = (Segment**)malloc(sizeof(Segment*) * segments.size());
	int i = 0;
	for(std::list<Segment*>::const_iterator it = segments.begin(); it != segments.end(); it++) {
		list[i] = *it;
		i++;
	}

	SegmentArray* segmentArray = (SegmentArray*)malloc(sizeof(SegmentArray));
	segmentArray->segments = list;
	segmentArray->size = segments.size();
	return segmentArray;
}

Segment* CpuRegionGrowingSegmentator::performNonRecursiveSegmentation(uchar* data, int channels,
		int step, int width, int height, int x, int y) {
	Segment* segment = createSegment(x, y);
	std::list<Point> points;
	points.push_back(createPoint(x, y));

	while(!points.empty()) {
		Point current = points.front();
		points.pop_front();
		if(isApplicable(data, channels, step, current.x, current.y)) {
			if(current.x > 0) points.push_back(createPoint(current.x - 1, current.y));
			if(current.x < segment->leftX) expandLeft(segment);

			if(current.y < width - 1) points.push_back(createPoint(current.x + 1, current.y));
			if(current.x > segment->rightX) expandRight(segment);

			if(current.y > 0) points.push_back(createPoint(current.x, current.y - 1));
			if(current.y < segment->bottomY) expandBottom(segment);

			if(current.y < height - 1) points.push_back(createPoint(current.x, current.y + 1));
			if(current.y > segment->topY) expandTop(segment);

			setNotApplicable(data, channels, step, current.x, current.y);
		}
	}

	return segment;
}

void CpuRegionGrowingSegmentator::setNotApplicable(uchar* data, int channels, int step, int x, int y) {
	int pos = x * channels + y * step;

	data[pos] = 0;
	data[pos+1] = 0;
	data[pos+2] = 0;
}

bool CpuRegionGrowingSegmentator::isApplicable(uchar* data, int channels, int step, int x, int y) {
	int pos = x * channels + y * step;

	int hue = data[pos];
	int sat = data[pos+1];
	int value = data[pos+2];

	if(sat != 0 && value != 0) {
		return true;
	}

	return false;
}

}}
