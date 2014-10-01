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

// helper structures
struct point {
	int x;
	int y;
};

struct element {
	struct point point;
	int next;
	int prev;
	int id;
	int v; // TODO
};

struct elements_pair {
	int id1;
	int id2;
};

}}
#endif
