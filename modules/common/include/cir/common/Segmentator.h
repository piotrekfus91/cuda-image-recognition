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

	virtual void init(int width, int height);
	virtual void shutdown();

	virtual void setMinSize(int size) = 0;

protected:
	int _minSize;
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
	bool valid;
	int v; // TODO
};

struct elements_pair {
	int id1;
	int id2;
};

}}
#endif
