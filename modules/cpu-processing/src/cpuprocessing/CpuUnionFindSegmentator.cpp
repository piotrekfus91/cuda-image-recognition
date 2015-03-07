#include "cir/cpuprocessing/CpuUnionFindSegmentator.h"
#include <iostream>
#include <iomanip>
#include <climits>
#include <list>

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuUnionFindSegmentator::CpuUnionFindSegmentator() : _segments(NULL), _ids(NULL) {

}

CpuUnionFindSegmentator::~CpuUnionFindSegmentator() {

}

void CpuUnionFindSegmentator::init(int width, int height) {
	_segments = (Segment*) malloc(sizeof(Segment) * width * height);
	_ids = (int*) malloc(sizeof(int) * width * height);
}

void CpuUnionFindSegmentator::shutdown() {
	if(_segments != NULL)
		free(_segments);

	if(_ids != NULL)
		free(_ids);
}

void CpuUnionFindSegmentator::initInternalStructures(uchar* data, int width, int height, int step, int channels) {
	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			int pos = x + width * y;
			int dataPos = x * channels + step * y;
			int idToSet = pos;
			if(channels == 1) {
				if(data[dataPos] == 0) {
					idToSet = -1;
				}
			} else if(channels == 3) {
				int saturation = data[dataPos+1];
				int value = data[dataPos+2];
				if(saturation == 0 && value == 0) {
					idToSet = -1;
				}
			}

			_ids[pos] = idToSet;
			_segments[pos] = createSimpleSegment(x, y);
		}
	}
}

SegmentArray* CpuUnionFindSegmentator::segmentate(const MatWrapper& input) {
	cv::Mat mat = input.getMat();
	int width = mat.cols;
	int height = mat.rows;
	int step = mat.step;
	int channels = mat.channels();
	uchar* data = mat.data;

	initInternalStructures(data, width, height, step, channels);

	bool changed = true;
	while(changed) {
		changed = false;
		unionFindSegmentate(width, height, step, channels, &changed);
	}

	std::list<Segment*> appliedSegments;
	int total = 0;
	for(int i = 0; i < width*height; i++) {
		if(i == _ids[i]) {
			Segment* segm = &_segments[i];
			if(isSegmentApplicable(segm)) {
				total++;
				appliedSegments.push_back(segm);
			}
		}
	}

	SegmentArray* segmentArray = (SegmentArray*) malloc(sizeof(SegmentArray));
	if(total > 0) {
		int idx = 0;
		Segment** segments = (Segment**) malloc(sizeof(Segment*) * total);
		for(std::list<Segment*>::iterator it = appliedSegments.begin(); it != appliedSegments.end(); it++) {
			Segment* copy = copySegment(*it);
			segments[idx] = copy;
			idx++;
		}
		segmentArray->segments = segments;
		segmentArray->size = total;
	} else {
		segmentArray->size = 0;
		segmentArray->segments = NULL;
	}

	return segmentArray;
}

void CpuUnionFindSegmentator::unionFindSegmentate(int width, int height, int step, int channels, bool* changed) {
	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			prepareBestNeighbour(x, y, width, height, changed);
		}
	}

	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			int pos = countPos(x, y, width, height);
			int currentId = _ids[pos];
			if(currentId != -1 && currentId != pos) {
				_ids[pos] = findRoot(pos);
			}
		}
	}
}

void CpuUnionFindSegmentator::prepareBestNeighbour(int x, int y, int width, int height, bool* changed) {
	int pos = countPos(x, y, width, height);
	if(_ids[pos] == -1)
		return;

	if(x > 0) {
		int neighbourPos = pos - 1;
		unite(pos, neighbourPos, changed);
	}

	if(x < width - 1) {
		int neighbourPos = pos + 1;
		unite(pos, neighbourPos, changed);
	}

	if(y > 0) {
		int neighbourPos = pos - width;
		unite(pos, neighbourPos, changed);
	}

	if(y < height - 1) {
		int neighbourPos = pos + width;
		unite(pos, neighbourPos, changed);
	}
}

int CpuUnionFindSegmentator::findRoot(int pos) {
	while(_ids[pos] != pos) {
		pos = _ids[pos];
	}
	return pos;
}

void CpuUnionFindSegmentator::unite(int pos1, int pos2, bool* changed) {
	int id1 = _ids[pos1];
	if(id1 == -1)
		return;

	int id2 = _ids[pos2];
	if(id2 == -1)
		return;

	int root1 = findRoot(pos1);
	int root2 = findRoot(pos2);

	if(root1 < root2) {
		mergeSegments(&_segments[_ids[root1]], &_segments[_ids[root2]]);
		_ids[root2] = root1;
		*changed = true;
	} else if(root1 > root2) {
		mergeSegments(&_segments[_ids[root1]], &_segments[_ids[root2]]);
		_ids[root1] = root2;
		*changed = true;
	}
}

int CpuUnionFindSegmentator::countPos(int x, int y, int width, int height) {
	return x + width * y;
}

}}