#ifndef GPUREGIONSPLITTINGSEGMENTATOR_H_
#define GPUREGIONSPLITTINGSEGMENTATOR_H_

#include "cir/common/Segmentator.h"
#include "cir/common/Segment.h"
#include "cir/common/SegmentArray.h"

namespace cir { namespace gpuprocessing {

class GpuRegionSplittingSegmentator : public cir::common::Segmentator {
public:
	GpuRegionSplittingSegmentator();
	virtual ~GpuRegionSplittingSegmentator();
	void init(int width, int height);

	virtual void setMinSize(int size);

	virtual cir::common::SegmentArray* segmentate(const cir::common::MatWrapper& input);
};

}}
#endif
