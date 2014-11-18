#include "cir/common/Segmentator.h"
#include "cir/common/config.h"
#include <cmath>
#include <iostream>

namespace cir { namespace common {

Segmentator::Segmentator() {
	_minSize = SEGMENTATOR_MIN_SIZE;
}

Segmentator::~Segmentator() {

}

void Segmentator::init(int width, int height) {

}

void Segmentator::shutdown() {

}

void Segmentator::setMinSize(int minSize) {
	_minSize = minSize;
}

bool Segmentator::isSegmentApplicable(Segment* segment) {
	int width = abs(segment->rightX - segment->leftX);
	int height = abs(segment->topY - segment->bottomY);
	return width >= _minSize && height >= _minSize;
}

}}
