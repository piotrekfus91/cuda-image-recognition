#ifndef GPUUNIONFINDSEGMENTATOR_H_
#define GPUUNIONFINDSEGMENTATOR_H_

#include "cir/common/Segmentator.h"

namespace cir { namespace gpuprocessing {

class GpuUnionFindSegmentator : public cir::common::Segmentator {
public:
	GpuUnionFindSegmentator();
	virtual ~GpuUnionFindSegmentator();

	void init(int width, int height);

	virtual void setMinSize(int size);

	virtual cir::common::SegmentArray* segmentate(const cir::common::MatWrapper& input);
};

}}
#endif /* GPUUNIONFINDSEGMENTATOR_H_ */
