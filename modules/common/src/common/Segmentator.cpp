#include "cir/common/Segmentator.h"

namespace cir { namespace common {

Segmentator::Segmentator() {

}

Segmentator::~Segmentator() {

}

bool Segmentator::isSegmentApplicable(Segment* segment) {
	int width = segment->rightX - segment->leftX;
	int height = segment->topY - segment->bottomY;
	return width > 20 && height > 20; // TODO config
}

}}
