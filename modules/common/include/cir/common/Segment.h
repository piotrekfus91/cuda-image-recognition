#ifndef SEGMENT_H_
#define SEGMENT_H_

namespace cir { namespace common {

struct Segment {
	int leftX;
	int rightX;
	int topY;
	int bottomY;
};

Segment createSimpleSegment(int x, int y);
Segment* createSegment(int x, int y);
Segment* copySegment(Segment* segment);
void mergeSegments(Segment* segm1, Segment* segm2);
void expandLeft(Segment* segment);
void expandRight(Segment* segment);
void expandTop(Segment* segment);
void expandBottom(Segment* segment);

}}
#endif
