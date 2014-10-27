#ifndef CPUREDMARKER_H_
#define CPUREDMARKER_H_

#include "cir/common/Marker.h"

namespace cir { namespace cpuprocessing {

class CpuRedMarker : public cir::common::Marker {
public:
	CpuRedMarker();
	virtual ~CpuRedMarker();

	cir::common::MatWrapper markSegments(cir::common::MatWrapper& input,
			const cir::common::SegmentArray* segmentArray);
};

}}
#endif
