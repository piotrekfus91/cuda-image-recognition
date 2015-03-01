#ifndef UNION_FIND_SEGMENTATE_CUH_
#define UNION_FIND_SEGMENTATE_CUH_

#include <vector_types.h>
#include <opencv2/core/types_c.h>
#include "cir/common/SegmentArray.h"

namespace cir { namespace gpuprocessing {

void union_find_segmentate_init(int width, int height);

cir::common::SegmentArray* union_find_segmentate(uchar* data, int step, int channels,
		int width, int height);

void union_find_segmentate_shutdown();

__global__
void k_init_internal_structures(uchar* data, int width, int height, int step, int channels,
		cir::common::Segment* segments, int* ids);

__global__
void k_prepare_best_neighbour(int* ids, cir::common::Segment* segments, int width, int height, bool* changed);

__global__
void k_find_best_root(int* ids, int width, int height);

__device__
int d_find_root(int* ids, int pos);

__device__
void d_unite(int pos1, int pos2, int* ids, cir::common::Segment* segments, bool* changed);

__device__ __host__
int d_count_pos(int x, int y, int width, int height);

__device__ __host__
void d_is_segment_applicable(cir::common::Segment* segment, bool* is_applicable, int min_size);

}}

#endif /* UNION_FIND_SEGMENTATE_CUH_ */
