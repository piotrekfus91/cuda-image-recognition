#ifndef REGION_SPLITTING_SEGMENTATE_H_
#define REGION_SPLITTING_SEGMENTATE_H_

#include <vector_types.h>
#include <opencv2/core/types_c.h>
#include "cir/common/SegmentArray.h"

namespace cir { namespace gpuprocessing {

struct point {
	int x;
	int y;
};

struct element {
	struct point point;
	int next;
	int prev;
	int id;
	int v; // TODO
};

struct elements_pair {
	int id1;
	int id2;
};

void region_splitting_segmentate_init(int width, int height);

void region_splitting_segmentate(uchar* data, int step, int channels, int width, int height);

void region_splitting_segmentate_shutdown();

__global__
void k_region_splitting_segmentate(uchar* data, elements_pair* merged_y,
		elements_pair* merged_x, element* elements, int step, int channels, int width, int height);

__device__
void d_merge_blocks_horizontally(int di_lb_top_right_x, int step, int channels,
		int ai_x, int width, int height, int ai_y, int merged_y_start_idx,
		int* merged_y_current_idx, uchar* data, element* elements,
		elements_pair* merged_y);

__device__
void d_merge_blocks_vertically(int di_lb_bottom_left_y, int step, int channels,
		int ai_x, int width, int height, int ai_y, int merged_x_start_idx,
		int *merged_x_current_idx, uchar* data, element* elements,
		elements_pair* merged_x);

__device__
void d_merge_elements(element* elements, element* e1, element* e2, int width);

__device__
bool d_is_empty(uchar* data, int addr);

__device__
bool d_already_merged(elements_pair* merged, int merged_start_idx, int merged_last_idx,
		element* e1, element* e2);

}}

#endif
