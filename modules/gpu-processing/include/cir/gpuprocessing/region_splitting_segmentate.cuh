#ifndef REGION_SPLITTING_SEGMENTATE_H_
#define REGION_SPLITTING_SEGMENTATE_H_

#include <vector_types.h>
#include <opencv2/core/types_c.h>
#include "cir/common/SegmentArray.h"
#include "cir/common/Segmentator.h"

namespace cir { namespace gpuprocessing {

/**
 * Performs initialization of segmentation.
 * The most important thing is allocation in RAM memory as well as GPU memory.
 *
 * @param width image width in pixels.
 * @param image height in pixels.
 */
void region_splitting_segmentate_init(int width, int height);

/**
 * Sets the minimum size (in pixels) of segment.
 * If segment have to be accepted, its width and height must exceed minimum size.
 *
 * @param minSize minimum segment size.
 */
void set_min_segment_size(int minSize);

/**
 * Performs segmentation using region splitting algorithm.
 * Returns array of accepted segments.
 *
 * @param data memory data, using uchar as data type. Must be in HSV model.
 * @param step OpenCV step ({@see http://docs.opencv.org/modules/core/doc/basic_structures.html#mat-mat}).
 * @param channels number of channels (currently only 3 channels in HSV model are supported).
 * @param width image width in pixels.
 * @param height image height in pixels.
 */
cir::common::SegmentArray* region_splitting_segmentate(uchar* data, int step, int channels, int width, int height);

/**
 * Performs shutdown.
 * Generally, frees previously allocated memory.
 */
void region_splitting_segmentate_shutdown();

/**
 * Kernel segmentation function.
 */
template <int block_width, int block_height>
__global__
void k_region_splitting_segmentate(cir::common::element* elements, cir::common::Segment* segments, int step,
		int channels, int width, int height);

/**
 * Removes empty segments.
 * Every empty pixel becomes a background, so its segment is marked as invalid.
 */
__global__
void k_remove_empty_segments(uchar* data, int width, int height, int step, cir::common::element* elements);

/**
 * Counts segments marked as accepted (valid and large enough to meet minimum segment size condition).
 */
__global__
void k_count_applicable_segments(cir::common::element* elements, cir::common::Segment* segments,
		int total_size, int min_size, int* partialSums);

/**
 * Merges two blocks horizontally.
 * Function goes from top to bottom pixels of neighbouring blocks and checks merge condition.
 * If both pixels meets condition, it merges its segment.
 */
template <int block_width, int block_height>
__device__
void d_merge_blocks_horizontally(int step, int channels, int ai_x, int width, int height, int ai_y,
		cir::common::element* elements, cir::common::Segment* segments);

/**
 * Merges two blocks vertically.
 * Performs action like {@link d_merge_blocks_horizontally}, but goes from left to right.
 */
template <int block_width, int block_height>
__device__
void d_merge_blocks_vertically(int step, int channels, int ai_x, int width, int height, int ai_y,
		cir::common::element* elements, cir::common::Segment* segments);

/**
 * Checks if two can be merged.
 * If yes, merges second segment to first, and invalidates second.
 */
__device__
void d_try_merge(int idx, int current_elem_id, int id_to_set, int width, int height,
		cir::common::element* elements, cir::common::Segment* segments, bool invalidate_all = true);

/**
 * Checks if pixel is empty (marked as background).
 */
__device__
bool d_is_empty(uchar* data, int addr);

/**
 * Performs segment boundaries update.
 */
__device__
void d_merge_segments(cir::common::Segment* segm1, cir::common::Segment* segm2);

/**
 * Checks if segment is applicable (large enough ({@link set_min_segment_size}) and valid).
 */
__device__ __host__
void d_is_segment_applicable(cir::common::Segment* segment, bool* is_applicable, int min_size);

}}

#endif
