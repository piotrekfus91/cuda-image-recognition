#ifndef CPUREGIONSPLITTINGSEGMENTATOR_H_
#define CPUREGIONSPLITTINGSEGMENTATOR_H_

#include "cir/common/Segmentator.h"
#include "cir/common/Segment.h"
#include "cir/common/SegmentArray.h"

namespace cir { namespace cpuprocessing {

class CpuRegionSplittingSegmentator : public cir::common::Segmentator {
public:
	CpuRegionSplittingSegmentator();
	virtual ~CpuRegionSplittingSegmentator();

	void init(int width, int height);
	void shutdown();

	virtual cir::common::SegmentArray* segmentate(const cir::common::MatWrapper& input);

private:
	cir::common::element* _elements;
	cir::common::Segment* _segments;


	void region_splitting_segmentate(uchar* data, cir::common::element* elements, cir::common::Segment* segments, int step,
			int channels, int width, int height, int ai_x, int ai_y, int block_width, int block_height);

	void remove_empty_segments(uchar* data, int width, int height, int step, cir::common::element* elements);

	void merge_blocks_horizontally(int di_lb_top_right_x, int step,
			int channels, int ai_x, int width, int height, int ai_y, uchar* data,
			cir::common::element* elements, cir::common::Segment* segments, int block_width, int block_height);

	void merge_blocks_vertically(int di_lb_bottom_left_y, int step,
			int channels, int ai_x, int width, int height, int ai_y, uchar* data,
			cir::common::element* elements, cir::common::Segment* segments, int block_width, int block_height);

	void try_merge(int idx, int current_elem_id, int id_to_set, int width, int height, bool invalidate_all = true);

	bool is_empty(uchar* data, int addr);

	void merge_segments(cir::common::Segment* segm1, cir::common::Segment* segm2);
};

}}
#endif
