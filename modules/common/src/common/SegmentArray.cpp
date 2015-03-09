#include "cir/common/SegmentArray.h"
#include <cstdlib>

namespace cir { namespace common {

SegmentArray* createSegmentArray(std::list<Segment*>& appliedSegments) {
	int size = appliedSegments.size();
	SegmentArray* segmentArray = (SegmentArray*) malloc(sizeof(SegmentArray));
	if(size > 0) {
		int idx = 0;
		Segment** segments = (Segment**) malloc(sizeof(Segment*) * size);
		for(std::list<Segment*>::iterator it = appliedSegments.begin(); it != appliedSegments.end(); it++) {
			Segment* copy = copySegment(*it);
			segments[idx] = copy;
			idx++;
		}
		segmentArray->segments = segments;
		segmentArray->size = size;
	} else {
		segmentArray->size = 0;
		segmentArray->segments = NULL;
	}

	return segmentArray;
}

void release(SegmentArray* segmentArray) {
	if(segmentArray == NULL)
		return;

	Segment** segments = segmentArray->segments;
	for(int i = 0; i < segmentArray->size; i++) {
		Segment* segment = segments[i];
		release(segment);
	}
	free(segmentArray);
}

}}
