#include <iostream>
#include <iomanip>
#include "cir/cpuprocessing/CpuRegionSplittingSegmentator.h"

// only HSV model supported so far
#define CHANNELS 3

using namespace cir::common;
using namespace std;

namespace cir { namespace cpuprocessing {

CpuRegionSplittingSegmentator::CpuRegionSplittingSegmentator()
		: _elements(NULL), _segments(NULL) {

}

CpuRegionSplittingSegmentator::~CpuRegionSplittingSegmentator() {

}

void CpuRegionSplittingSegmentator::init(int width, int height) {
	_elements = (element*) malloc(sizeof(element) * width * height);
	_segments = (Segment*) malloc(sizeof(Segment) * width * height);
}

void CpuRegionSplittingSegmentator::setMinSize(int size) {
	_minSize = size;
}

void CpuRegionSplittingSegmentator::remove_empty_segments(uchar* data, int width, int height, int step, element* elements) {
	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			int pos = x * CHANNELS + y * step;
			uchar saturation = data[pos+1];
			uchar value = data[pos+2];

			if(saturation == 0 && value == 0) {
				element* elem = &(elements[x + width * y]);
				elem->id = -1;
				elem->valid = false;
			}
		}
	}
}

SegmentArray* CpuRegionSplittingSegmentator::segmentate(const MatWrapper& input) {
	cv::Mat mat = input.getMat();
	int width = mat.cols;
	int height = mat.rows;
	int step = mat.step;
	uchar* data = mat.data;

	for(int i = 0; i < width*height; i++) {
		_elements[i].id = i;
		_elements[i].next = i;
		_elements[i].prev = i;
		_elements[i].point.x = i % width;
		_elements[i].point.y = i / width;
		_elements[i].v = i;
		_elements[i].valid = true;

		_segments[i] = createSimpleSegment(i % width, i / width);
	}

	remove_empty_segments(data, width, height, step, _elements);

	int greaterDim = width > height ? width : height;

	// divide and conquer
	for(int i = 1; i < greaterDim; i = 2 * i) {

		for(int x = 0; x < width; x += i) {
			for(int y = 0; y < height; y+= i) {
				int block_width = i;
				int block_height = i;

				region_splitting_segmentate(data, _elements, _segments, step,
						CHANNELS, width, height, x, y, block_width, block_height);

			}
		}
	}

	int foundSegmentsSize = 0;
	for(int j = 0; j < width*height; j++) {
		if(_elements[j].valid && isSegmentApplicable(&(_segments[j])))
			foundSegmentsSize++;
	}

	SegmentArray* segmentArray = (SegmentArray*) malloc(sizeof(SegmentArray));
	segmentArray->size = foundSegmentsSize;

	if(foundSegmentsSize > 0) {
		Segment** segments = (Segment**) malloc(sizeof(Segment*) * foundSegmentsSize);
		int currentSegmentIndex = 0;
		for(int j = 0; j < width*height; j++) {
			if(_elements[j].valid && isSegmentApplicable(&(_segments[j]))) {
				Segment segment = _segments[j];
				segments[currentSegmentIndex++] = copySegment(&segment);
			}
		}
		segmentArray->segments = segments;
	} else {
		segmentArray->segments = NULL;
	}

	return segmentArray;
}

void CpuRegionSplittingSegmentator::shutdown() {
	free(_elements);
	free(_segments);
}

void CpuRegionSplittingSegmentator::region_splitting_segmentate(uchar* data, element* elements, Segment* segments,
		int step, int channels, int width, int height, int ai_x, int ai_y, int block_width, int block_height) {
	if(ai_x % (block_width * 2) != 0 || ai_y % (block_height * 2) != 0)
		return;

	// top left and top right
	int di_tlb_top_right_x = (ai_x + block_width - 1) * channels + ai_y * step;
	int ai_lb_top_right_x = ai_x + block_width - 1;

	merge_blocks_horizontally(di_tlb_top_right_x, step, channels, ai_lb_top_right_x, width, height,
			ai_y, data, elements, segments, block_width, block_height);

	// bottom left and bottom right
	int di_blb_top_right_x = di_tlb_top_right_x + block_height * step;
	int blb_ai_y = ai_y + block_height;

	merge_blocks_horizontally(di_blb_top_right_x, step, channels, ai_lb_top_right_x, width, height,
			blb_ai_y, data, elements, segments, block_width, block_height);

	// top left/right and bottom left/right
	int di_tb_bottom_left_y = ai_x * channels + (ai_y + block_height - 1) * step;
	merge_blocks_vertically(di_tb_bottom_left_y, step, channels, ai_x, width, height, ai_y + block_height - 1,
			data, elements, segments, block_width, block_height);
}

void CpuRegionSplittingSegmentator::merge_blocks_horizontally(int di_lb_top_right_x, int step,
		int channels, int ai_x, int width, int height, int ai_y, uchar* data, element* elements,
		Segment* segments, int block_width, int block_height) {

	for (int i = 0; i < block_height; i++) {
		int ai_tlb = ai_x + width * (i + ai_y);
		int ai_trb = ai_tlb + 1;

		if(ai_trb % width < ai_tlb % width || ai_trb > width * height)
			return;

		element* left_elem = &(elements[ai_tlb]);
		element* right_elem = &(elements[ai_trb]);

		int left_elem_id = left_elem->id;
		int right_elem_id = right_elem->id;

		if (left_elem_id != -1 && right_elem_id != -1) {

			for(int j = 0; j < block_height; j++) {
				int ai_tlb_right = ai_x + width * j + ai_y * width;
				try_merge(ai_tlb_right, right_elem_id, left_elem_id, width, height, false);

				int ai_trb_left = ai_tlb_right + 1;
				try_merge(ai_trb_left, right_elem_id, left_elem_id, width, height);

				int x_trb_left = ai_trb_left % width;
				int normalized_width = x_trb_left + block_width - 1 > width ? width - x_trb_left + 1 : block_width;

				int ai_trb_right = ai_trb_left + normalized_width - 1;
				try_merge(ai_trb_right, right_elem_id, left_elem_id, width, height);

				int ai_tlb_left = ai_trb_left - block_width;
				try_merge(ai_tlb_left, right_elem_id, left_elem_id, width, height, false);
			}

			for(int j = 0; j < block_width; j++) {
				int ai_trb_top = ai_x + j + 1 + ai_y * width;
				try_merge(ai_trb_top, right_elem_id, left_elem_id, width, height);

				int ai_tlb_top = ai_trb_top - block_width;
				try_merge(ai_tlb_top, right_elem_id, left_elem_id, width, height, false);

				int y_trb_top = ai_trb_top / width;
				int normalized_height = y_trb_top + block_height - 1 > height ? height - y_trb_top + 1 : block_height;

				int ai_trb_bottom = (normalized_height - 1) * width + ai_trb_top;
				try_merge(ai_trb_bottom, right_elem_id, left_elem_id, width, height);

				int ai_tlb_bottom = ai_trb_bottom - block_width;
				try_merge(ai_tlb_bottom, right_elem_id, left_elem_id, width, height, false);
			}
		}
	}
}

void CpuRegionSplittingSegmentator::merge_blocks_vertically(int di_lb_bottom_left_y, int step,
		int channels, int ai_x, int width, int height, int ai_y, uchar* data, element* elements,
		Segment* segments, int block_width, int block_height) {

	for (int i = 0; i < 2*block_width; i++) {
		int ai_tb = ai_x + i + width * ai_y;
		int ai_bb = ai_tb + width;

		if(ai_bb / width > height || ai_bb > width * height)
			return;

		element* top_elem = &(elements[ai_tb]);
		element* bottom_elem = &(elements[ai_bb]);

		int top_elem_id = top_elem->id;
		int bottom_elem_id = bottom_elem->id;

		if (top_elem_id != -1 && bottom_elem_id != -1) {

			for(int j = 0; j < 2*block_width; j++) {
				int ai_bb_top = ai_x + width + j + ai_y * width;
				try_merge(ai_bb_top, bottom_elem_id, top_elem_id, width, height);

				int ai_tb_top = ai_bb_top - block_height * width;
				try_merge(ai_tb_top, bottom_elem_id, top_elem_id, width, height, false);

				int y_bb_top = ai_bb_top / width;
				int normalized_height = y_bb_top + block_height - 1 > height ? height - y_bb_top + 1 : block_height;

				int ai_bb_bottom = ai_bb_top + (normalized_height-1) * width;
				try_merge(ai_bb_bottom, bottom_elem_id, top_elem_id, width, height);

				int ai_tb_bottom = ai_bb_bottom - block_height * width;
				try_merge(ai_tb_bottom, bottom_elem_id, top_elem_id, width, height, false);
			}

			for(int j = 0; j < block_height; j++) {
				int ai_bb_left = ai_x + (j+1) * width + ai_y * width;
				try_merge(ai_bb_left, bottom_elem_id, top_elem_id, width, height);

				int ai_tb_left = ai_bb_left - block_height * width;
				try_merge(ai_tb_left, bottom_elem_id, top_elem_id, width, height, false);

				int x_bb_left = ai_bb_left % width;
				int normalized_width = x_bb_left + 2*block_width - 1 > width ? width - x_bb_left + 1 : 2 * block_width;

				int ai_bb_right = ai_bb_left + normalized_width - 1;
				try_merge(ai_bb_right, bottom_elem_id, top_elem_id, width, height);

				int ai_tb_right = ai_bb_right - block_height * width;
				try_merge(ai_tb_right, bottom_elem_id, top_elem_id, width, height, false);
			}
		}
	}
}

void CpuRegionSplittingSegmentator::try_merge(int idx, int current_elem_id, int id_to_set, int width, int height, bool invalidate_all) {
	if(idx < width * height) {
		element* elem = &(_elements[idx]);
		if(elem->id == current_elem_id) {
			Segment* segm1 = &(_segments[id_to_set]);
			Segment* segm2 = &(_segments[elem->id]);
			if(id_to_set != elem->id) {
				element* elemToInvalidate = &(_elements[elem->id]);
				if(invalidate_all)
					elemToInvalidate->valid = false;
				elemToInvalidate->id = id_to_set;
			}
			merge_segments(segm1, segm2);
			if(invalidate_all)
				elem->valid = false;
			elem->id = id_to_set;
		}
	}
}

bool CpuRegionSplittingSegmentator::is_empty(uchar* data, int addr) {
	return data[addr+1] == 0 && data[addr+2] == 0;
}

void CpuRegionSplittingSegmentator::merge_segments(Segment* segm1, Segment* segm2) {
	if(segm1->leftX < segm2->leftX) {
		segm2->leftX = segm1->leftX;
	} else {
		segm1->leftX = segm2->leftX;
	}

	if(segm1->rightX > segm2->rightX) {
		segm2->rightX = segm1->rightX;
	} else {
		segm1->rightX = segm2->rightX;
	}

	if(segm1->bottomY > segm2->bottomY) {
		segm2->bottomY = segm1->bottomY;
	} else {
		segm1->bottomY = segm2->bottomY;
	}

	if(segm1->topY < segm2->topY) {
		segm2->topY = segm1->topY;
	} else {
		segm1->topY = segm2->topY;
	}
}

}}
