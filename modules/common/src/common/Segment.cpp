#include <cstdlib>
#include "cir/common/Segment.h"

namespace cir { namespace common {

Segment* createSegment(int x, int y) {
	Segment* segment = (Segment*)malloc(sizeof(Segment));
	segment->leftX = x;
	segment->rightX = x;
	segment->topY = y;
	segment->bottomY = y;
	return segment;
}

void expandLeft(Segment* segment) {
	segment->leftX--;
}

void expandRight(Segment* segment) {
	segment->rightX++;
}

void expandTop(Segment* segment) {
	segment->topY++;
}

void expandBottom(Segment* segment) {
	segment->bottomY--;
}

}}
