#include <iostream>
#include <iomanip>
#include "cir/cpuprocessing/CpuRegionSplittingSegmentator.h"

// only HSV model supported so far
#define CHANNELS 3

using namespace cir::common;
using namespace std;

namespace cir { namespace cpuprocessing {

CpuRegionSplittingSegmentator::CpuRegionSplittingSegmentator()
		: _elements(NULL), _merged_x(NULL), _merged_y(NULL) {

}

CpuRegionSplittingSegmentator::~CpuRegionSplittingSegmentator() {

}

void CpuRegionSplittingSegmentator::init(int width, int height) {
	_elements = (element*) malloc(sizeof(element) * width * height);

	_merged_y = (elements_pair*) malloc(sizeof(elements_pair) * width * height);
	_merged_x = (elements_pair*) malloc(sizeof(elements_pair) * width * height);
}

SegmentArray* CpuRegionSplittingSegmentator::segmentate(const MatWrapper& input) {
	cv::Mat mat = input.getMat();
	int width = mat.cols;
	int height = mat.rows;

	for(int i = 0; i < width*height; i++) {
		_elements[i].id = i;
		_elements[i].next = i;
		_elements[i].prev = i;
		_elements[i].point.x = i % width;
		_elements[i].point.y = i / width;
		_elements[i].v = i;

		_merged_x[i].id1 = -1;
		_merged_x[i].id2 = -1;
		_merged_y[i].id1 = -1;
		_merged_y[i].id2 = -1;
	}

	int greaterDim = width > height ? width : height;

	// divide and conquer
	for(int i = 1; i < greaterDim; i = 2 * i) {

		for(int x = 0; x < width; x += i) {
			for(int y = 0; y < height; y+= i) {
				int block_width = i;
				int block_height = i;
				region_splitting_segmentate(mat.data, _merged_y, _merged_x, _elements, mat.step,
						CHANNELS, width, height, x, y, block_width, block_height);

			}
		}

//		for(int x = 0; x < width; x++) {
//			for(int y = 0; y < height; y++) {
//				cerr << setw(3) << _elements[x*height + y].id;
//			}
//			cerr << endl;
//		}
//
//		cerr << "===========================" << endl;
	}

	return NULL;
}

void CpuRegionSplittingSegmentator::shutdown() {
	free(_elements);
	free(_merged_x);
	free(_merged_y);
}

void CpuRegionSplittingSegmentator::region_splitting_segmentate(uchar* data, elements_pair* merged_y,
		elements_pair* merged_x, element* elements, int step, int channels, int width,
		int height, int ai_x, int ai_y, int block_width, int block_height) {
	if(ai_x % (block_width * 2) != 0 || ai_y % (block_height * 2) != 0)
		return;

	int merged_y_start_idx = ai_x + ai_y * block_width;
	int merged_y_current_idx = merged_y_start_idx;

	int merged_x_start_idx = ai_x + ai_y * block_width;
	int merged_x_current_idx = merged_x_start_idx;

	// top left and top right
	int di_tlb_top_right_x = (ai_x + block_width - 1) * channels + ai_y * step;
	int ai_lb_top_right_x = ai_x + block_width - 1;

	merge_blocks_horizontally(di_tlb_top_right_x, step, channels, ai_lb_top_right_x, width, height,
			ai_y, merged_y_start_idx, &merged_y_current_idx, data, elements,
			merged_y, block_height);

	// bottom left and bottom right
	int di_blb_top_right_x = di_tlb_top_right_x + block_height * step;
	int blb_ai_y = ai_y + block_height;

	merge_blocks_horizontally(di_blb_top_right_x, step, channels, ai_lb_top_right_x, width, height,
			blb_ai_y, merged_y_start_idx, &merged_y_current_idx, data, elements,
			merged_y, block_height);

	// top left/right and bottom left/right
	int di_tb_bottom_left_y = ai_x * channels + (ai_y + block_height - 1) * step;
	merge_blocks_vertically(di_tb_bottom_left_y, step, channels, ai_x, width, height, ai_y + block_height - 1,
			merged_x_start_idx, &merged_x_current_idx, data, elements, merged_x, block_width);
}

void CpuRegionSplittingSegmentator::merge_blocks_horizontally(int di_lb_top_right_x, int step,
		int channels, int ai_x, int width, int height, int ai_y, int merged_y_start_idx,
		int *merged_y_current_idx, uchar* data, element* elements,
		elements_pair* merged_y, int block_height) {

	for (int i = 0; i < block_height; i++) {
		int di_tlb_right = di_lb_top_right_x + i * step;
		int di_trb_left = di_tlb_right + channels;
		int ai_tlb = ai_x + width * (i + ai_y);
		int ai_trb = ai_tlb + 1;

		if(ai_trb % height > width || ai_trb > width * height)
			return;

		if (!is_empty(data, di_tlb_right) && !is_empty(data, di_trb_left)) {
			element* left_elem = &(elements[ai_tlb]);
			element* right_elem = &(elements[ai_trb]);
			if (already_merged(merged_y, merged_y_start_idx, *merged_y_current_idx, left_elem, right_elem))
				continue;

			merge_elements(elements, left_elem, right_elem, width);
			merged_y[*merged_y_current_idx].id1 = left_elem->id;
			merged_y[*merged_y_current_idx].id2 = right_elem->id;
			*merged_y_current_idx += 1;
		}
	}
}

void CpuRegionSplittingSegmentator::merge_blocks_vertically(int di_lb_bottom_left_y, int step,
		int channels, int ai_x, int width, int height, int ai_y, int merged_x_start_idx,
		int *merged_x_current_idx, uchar* data, element* elements,
		elements_pair* merged_x, int block_width) {

	for (int i = 0; i < 2*block_width; i++) {
		int di_tlb_bottom = di_lb_bottom_left_y + i * channels;
		int di_blb_top = di_tlb_bottom + step;
		int ai_tb = ai_x + i + width * ai_y;
		int ai_bb = ai_tb + width;

		if(ai_bb / width > height || ai_bb > width * height)
			return;

		if (!is_empty(data, di_tlb_bottom) && !is_empty(data, di_blb_top)) {
			element* top_elem = &(elements[ai_tb]);
			element* bottom_elem = &(elements[ai_bb]);
			if (already_merged(merged_x, merged_x_start_idx, *merged_x_current_idx, top_elem, bottom_elem))
				continue;

			merge_elements(elements, top_elem, bottom_elem, width);
			merged_x[*merged_x_current_idx].id1 = top_elem->id;
			merged_x[*merged_x_current_idx].id2 = bottom_elem->id;
			*merged_x_current_idx += 1;
		}
	}
}

void CpuRegionSplittingSegmentator::merge_elements(element* elements, element* e1, element* e2, int width) {
	(&(elements[e1->next]))->prev = e2->prev;
	(&(elements[e2->prev]))->next = e1->next;
	e1->next = width * e2->point.y + e2->point.x;
	e2->prev = width * e1->point.y + e1->point.x;

	e2->id = e1->id;

	// TODO very ineffective
	int end = elements[e2->next].prev; // converts element to its position
	for(int i = e2->next; i != end;) {
		element* elem = &(elements[i]);
		elem->id = e1->id;
		i = elem->next;
	}

}

bool CpuRegionSplittingSegmentator::is_empty(uchar* data, int addr) {
	return data[addr+1] == 0 && data[addr+2] == 0;
}

bool CpuRegionSplittingSegmentator::already_merged(elements_pair* merged, int merged_start_idx,
		int merged_last_idx, element* e1, element* e2) {
	for(int i = merged_start_idx; i < merged_last_idx; i++) {
		if(merged[i].id1 == e1->id && merged[i].id2 == e2->id)
			return true;
	}

	return false;
}

}}
