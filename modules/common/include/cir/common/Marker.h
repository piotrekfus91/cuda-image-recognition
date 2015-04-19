#ifndef MARKER_H_
#define MARKER_H_

#include "cir/common/MatWrapper.h"
#include "cir/common/SegmentArray.h"

namespace cir { namespace common {

class Marker {
public:
	Marker();
	virtual ~Marker();

	virtual MatWrapper markSegments(MatWrapper input, const SegmentArray* segmentArray) = 0;
	virtual MatWrapper markPairs(MatWrapper input,
			std::vector<std::pair<Segment*, int> > pairs) = 0;
};

}}
#endif
