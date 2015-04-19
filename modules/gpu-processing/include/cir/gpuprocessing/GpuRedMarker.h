#ifndef GPUREDMARKER_H_
#define GPUREDMARKER_H_

#include "cir/common/Marker.h"

namespace cir { namespace gpuprocessing {

class GpuRedMarker : public cir::common::Marker {
public:
	GpuRedMarker();
	virtual ~GpuRedMarker();

	cir::common::MatWrapper markSegments(cir::common::MatWrapper input,
			const cir::common::SegmentArray* segmentArray);

	virtual cir::common::MatWrapper markPairs(cir::common::MatWrapper input,
			std::vector<std::pair<cir::common::Segment*, int> > pairs);

protected:
	void mark(cir::common::MatWrapper& mw, cir::common::Segment* segment, int red, int green, int blue);
};

}}
#endif
