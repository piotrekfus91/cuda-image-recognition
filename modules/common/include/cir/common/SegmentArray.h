#ifndef SEGMENTLIST_H_
#define SEGMENTLIST_H_

#include "cir/common/Segment.h"

namespace cir { namespace common {

struct SegmentArray {
	Segment** segments;
	int size;
};

}}
#endif
