#ifndef SEGMENTLIST_H_
#define SEGMENTLIST_H_

#include <list>
#include "cir/common/Segment.h"

namespace cir { namespace common {

struct SegmentArray {
	Segment** segments;
	int size;
};

SegmentArray* createSegmentArray(std::list<Segment*>& segments);
void release(SegmentArray* segmentArray);

}}
#endif
