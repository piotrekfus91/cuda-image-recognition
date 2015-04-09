#ifndef REGION_SPLITTING_SEGMENTATE_H_
#define REGION_SPLITTING_SEGMENTATE_H_

#include <vector_types.h>
#include <opencv2/core/types_c.h>
#include "cir/common/SegmentArray.h"
#include "cir/common/Segmentator.h"

namespace cir { namespace gpuprocessing {

void region_splitting_segmentate_init(int width, int height);

cir::common::SegmentArray* region_splitting_segmentate(uchar* data, int step, int channels, int width, int height,
		cudaStream_t stream);

void region_splitting_segmentate_shutdown();

__global__
void k_region_splitting_segmentate(uchar* data, cir::common::element* elements, cir::common::Segment* segments, int step,
		int channels, int width, int height, int block_width, int block_height);

__global__
void k_remove_empty_segments(uchar* data, int width, int height, int step, cir::common::element* elements);

__global__
void k_count_applicable_segments(cir::common::element* elements, cir::common::Segment* segments,
		int total_size, int min_size, int* partialSums);

__device__
void d_merge_blocks_horizontally(int di_lb_top_right_x, int step,
		int channels, int ai_x, int width, int height, int ai_y, uchar* data,
		cir::common::element* elements, cir::common::Segment* segments, int block_width, int block_height);

__device__
void d_merge_blocks_vertically(int di_lb_bottom_left_y, int step,
		int channels, int ai_x, int width, int height, int ai_y, uchar* data,
		cir::common::element* elements, cir::common::Segment* segments, int block_width, int block_height);

__device__
void d_try_merge(int idx, int current_elem_id, int id_to_set, int width, int height,
		cir::common::element* elements, cir::common::Segment* segments, bool invalidate_all = true);

__device__
bool d_is_empty(uchar* data, int addr);

__device__ __host__
void d_is_segment_applicable_rs(cir::common::Segment* segment, bool* is_applicable, int min_size);

}}

#endif
