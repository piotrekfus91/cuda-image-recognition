#include <cstdlib>
#include "cir/common/Segment.h"

namespace cir { namespace common {

Segment createSimpleSegment(int x, int y) {
	Segment segment;
	segment.leftX = x;
	segment.rightX = x;
	segment.topY = y;
	segment.bottomY = y;
	return segment;
}

Segment* createSegment(int x, int y) {
	Segment* segment = (Segment*)malloc(sizeof(Segment));
	segment->leftX = x;
	segment->rightX = x;
	segment->topY = y;
	segment->bottomY = y;
	return segment;
}

Segment* copySegment(Segment* segment) {
	Segment* copy = (Segment*) malloc(sizeof(Segment));
	copy->leftX = segment->leftX;
	copy->rightX = segment->rightX;
	copy->topY = segment->topY;
	copy->bottomY = segment->bottomY;
	return copy;
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
