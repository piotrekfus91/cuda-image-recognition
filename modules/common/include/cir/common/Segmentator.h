#ifndef SEGMENTATOR_H_
#define SEGMENTATOR_H_

#include "cir/common/Segment.h"
#include "cir/common/SegmentArray.h"
#include "cir/common/MatWrapper.h"

namespace cir { namespace common {

class Segmentator {
public:
	Segmentator();
	virtual ~Segmentator();

	virtual SegmentArray* segmentate(const MatWrapper& matWrapper) = 0;

protected:
	bool isSegmentApplicable(Segment* segment);
};

}}
#endif
