#ifndef CPUREGIONGROWINGSEGMENTATOR_H_
#define CPUREGIONGROWINGSEGMENTATOR_H_

#include "cir/common/Segmentator.h"
#include "cir/common/Segment.h"
#include "cir/common/SegmentArray.h"

namespace cir { namespace cpuprocessing {

class CpuRegionGrowingSegmentator : public cir::common::Segmentator {
public:
	CpuRegionGrowingSegmentator();
	virtual ~CpuRegionGrowingSegmentator();

	virtual cir::common::SegmentArray* segmentate(const cir::common::MatWrapper& input);

private:
	cir::common::Segment* performNonRecursiveSegmentation(uchar* data, int channels,
			int step, int width, int height, int x, int y);
	void setNotApplicable(uchar* data, int channels, int step, int x, int y);
	bool isApplicable(uchar* data, int channels, int step, int x, int y);
};

}}
#endif
