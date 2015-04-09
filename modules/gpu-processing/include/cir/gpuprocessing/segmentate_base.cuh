#ifndef SEGMENTATE_BASE_CUH_
#define SEGMENTATE_BASE_CUH_

#include "cir/common/Segment.h"

namespace cir { namespace gpuprocessing {

extern int _min_size;

extern cir::common::Segment* segments;
extern cir::common::Segment* d_segments;

void set_segment_min_size(int size);

__device__ __host__
void d_merge_segments(cir::common::Segment* segm1, cir::common::Segment* segm2);

__device__
void d_merge_segments_rs(cir::common::Segment* segm1, cir::common::Segment* segm2);

}}

#endif /* SEGMENTATE_BASE_CUH_ */
