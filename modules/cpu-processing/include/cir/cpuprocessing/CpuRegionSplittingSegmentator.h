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
	cir::common::elements_pair* _merged_x;
	cir::common::elements_pair* _merged_y;


	void region_splitting_segmentate(uchar* data, cir::common::elements_pair* merged_y,
			cir::common::elements_pair* merged_x, cir::common::element* elements, int step,
			int channels, int width, int height, int ai_x, int ai_y, int block_width, int block_height);

	void merge_blocks_horizontally(int di_lb_top_right_x, int step,
			int channels, int ai_x, int width, int height, int ai_y, int merged_y_start_idx,
			int *merged_y_current_idx, uchar* data, cir::common::element* elements,
			cir::common::elements_pair* merged_y, int block_height);

	void merge_blocks_vertically(int di_lb_bottom_left_y, int step,
			int channels, int ai_x, int width, int height, int ai_y, int merged_x_start_idx,
			int *merged_x_current_idx, uchar* data, cir::common::element* elements,
			cir::common::elements_pair* merged_x, int block_width);

	void merge_elements(cir::common::element* elements, cir::common::element* e1,
			cir::common::element* e2, int width);

	bool is_empty(uchar* data, int addr);

	bool already_merged(cir::common::elements_pair* merged, int merged_start_idx,
			int merged_last_idx, cir::common::element* e1, cir::common::element* e2);
};

}}
#endif
