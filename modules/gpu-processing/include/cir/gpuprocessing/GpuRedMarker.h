#ifndef GPUREDMARKER_H_
#define GPUREDMARKER_H_

#include "cir/common/Marker.h"

namespace cir { namespace gpuprocessing {

class GpuRedMarker : public cir::common::Marker {
public:
	GpuRedMarker();
	virtual ~GpuRedMarker();

	cir::common::MatWrapper markSegments(cir::common::MatWrapper& input,
			const cir::common::SegmentArray* segmentArray);
};

}}
#endif
