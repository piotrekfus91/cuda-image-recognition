#include <cstdlib>
#include "cir/common/Segment.h"

namespace cir { namespace common {

bool Segment::contains(int x, int y) {
	return leftX <= x && x <= rightX && topY <= y && y <= bottomY;
}

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

void release(Segment* segment) {
	if(segment == NULL)
		return;

	free(segment);
}

void mergeSegments(Segment* segm1, Segment* segm2) {
	if(segm1->leftX < segm2->leftX) {
		segm2->leftX = segm1->leftX;
	} else {
		segm1->leftX = segm2->leftX;
	}

	if(segm1->rightX > segm2->rightX) {
		segm2->rightX = segm1->rightX;
	} else {
		segm1->rightX = segm2->rightX;
	}

	if(segm1->topY < segm2->topY) {
		segm2->topY = segm1->topY;
	} else {
		segm1->topY = segm2->topY;
	}

	if(segm1->bottomY > segm2->bottomY) {
		segm2->bottomY = segm1->bottomY;
	} else {
		segm1->bottomY = segm2->bottomY;
	}
}

void expandLeft(Segment* segment) {
	segment->leftX--;
}

void expandRight(Segment* segment) {
	segment->rightX++;
}

void expandTop(Segment* segment) {
	segment->topY--;
}

void expandBottom(Segment* segment) {
	segment->bottomY++;
}

}}
