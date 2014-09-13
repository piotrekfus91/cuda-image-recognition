#include <vector_types.h>
#include <iostream>
#include "cir/gpuprocessing/region_splitting_segmentate.cuh"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

void region_splitting_segmentate(uchar* data, int step, int channels, int width, int height) {
	element* elements = (element*)malloc(sizeof(element) * width * height);
	for(int i = 0; i < width*height; i++) {
		elements[i].id = i;
		elements[i].next = i;
		elements[i].prev = i;
		elements[i].point.x = i % width;
		elements[i].point.y = i / width;
		elements[i].v = i;
	}

	elements_pair* merged_y = (elements_pair*) malloc(sizeof(elements_pair) * width * height);
	elements_pair* merged_x = (elements_pair*) malloc(sizeof(elements_pair) * width * height);
	for(int i = 0; i < width*height; i++) {
		merged_x[i].id1 = -1;
		merged_y[i].id2 = -1;
	}

	element* d_elements;
	elements_pair* d_merged_y;
	elements_pair* d_merged_x;

	cudaMalloc((void**) &d_elements, sizeof(element) * width * height);
	cudaMalloc((void**) &d_merged_y, sizeof(elements_pair) * width * height);
	cudaMalloc((void**) &d_merged_x, sizeof(elements_pair) * width * height);

	cudaMemcpy(d_elements, elements, sizeof(element) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_merged_y, merged_y, sizeof(elements_pair) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_merged_x, merged_x, sizeof(elements_pair) * width * height, cudaMemcpyHostToDevice);

	dim3 init_blocks(width, height);

	int greaterDim = width > height ? width : height;

	for(int i = 1; i < greaterDim; i = 2 * i) {
		dim3 blocks((width+i-1)/i, (height+i-1)/i);
		dim3 threads(1, 1);
		k_region_splitting_segmentate<<<blocks, threads>>>(data, d_merged_y, d_merged_x, d_elements,
				step, channels, width, height);
		cudaMemcpy(elements, d_elements, sizeof(element) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(merged_x, d_merged_x, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(merged_y, d_merged_y, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
		for(int x = 0; x < width; x++) {
			for(int y = 0; y < height; y++) {
				std::cerr << elements[x*height + y].id << " ";
			}

			std::cerr << std::endl;
		}
		std::cerr << "-----------" << std::endl;
		cudaDeviceSynchronize();
	}

	cudaFree(d_elements);
	cudaFree(d_merged_y);
	cudaFree(d_merged_x);

	free(merged_y);
	free(merged_x);
	free(elements);
}

// tlb - top left block
// brb - bottom right block
// di_ - data index
// ai - array index
__global__
void k_region_splitting_segmentate(uchar* data, elements_pair* merged_y, elements_pair* merged_x,
		element* elements, int step, int channels, int width, int height) {
	int ai_x = blockIdx.x * blockDim.x + threadIdx.x;
	int ai_y = blockIdx.y * blockDim.y + threadIdx.y;

	if(ai_x % 2 != 0 || ai_y % 2 != 0)
		return;

	int block_width = width / gridDim.x;
	int block_height = height / gridDim.y;

	ai_x = ai_x * block_width;
	ai_y = ai_y * block_height;

	int merged_y_start_idx = ai_x + ai_y * blockDim.x * gridDim.x;
	int merged_y_current_idx = merged_y_start_idx;

	int merged_x_start_idx = ai_x + ai_y * blockDim.x * gridDim.x;
	int merged_x_current_idx = merged_x_start_idx;

	// top left and top right
	int di_tlb_top_right_x = (ai_x + block_width - 1) * channels + ai_y * step;
	int ai_lb_top_right_x = ai_x + block_width - 1;

	d_merge_blocks_horizontally(di_tlb_top_right_x, step, channels, ai_lb_top_right_x, width,
			ai_y, merged_y_start_idx, &merged_y_current_idx, data, elements,
			merged_y);

	// bottom left and bottom right
	int di_blb_top_right_x = di_tlb_top_right_x + block_height * step;
	int blb_ai_y = ai_y + block_height;

	d_merge_blocks_horizontally(di_blb_top_right_x, step, channels, ai_lb_top_right_x, width,
			blb_ai_y, merged_y_start_idx, &merged_y_current_idx, data, elements,
			merged_y);

	// top left/right and bottom left/right
	int di_tb_bottom_left_y = ai_x * channels + (ai_y + block_height - 1) * step;
	d_merge_blocks_vertically(di_tb_bottom_left_y, step, channels, ai_x, width, ai_y + block_height - 1,
			merged_x_start_idx, &merged_x_current_idx, data, elements, merged_x);
}

__device__
void d_merge_blocks_horizontally(int di_lb_top_right_x, int step, int channels,
		int ai_x, int width, int ai_y, int merged_y_start_idx,
		int *merged_y_current_idx, uchar* data, element* elements,
		elements_pair* merged_y) {

	int block_height = width / gridDim.x; // TODO

	for (int i = 0; i < block_height; i++) {
		int di_tlb_right = di_lb_top_right_x + i * step;
		int di_trb_left = di_tlb_right + channels;
		int ai_tlb = ai_x + width * (i + ai_y);
		int ai_trb = ai_tlb + 1;
		if (!d_is_empty(data, di_tlb_right) && !d_is_empty(data, di_trb_left)) {
			element* left_elem = &(elements[ai_tlb]);
			element* right_elem = &(elements[ai_trb]);
			if (d_already_merged(merged_y, merged_y_start_idx, *merged_y_current_idx, left_elem, right_elem))
				continue;

			d_merge_elements(elements, left_elem, right_elem, width);
			merged_y[*merged_y_current_idx].id1 = left_elem->id;
			merged_y[*merged_y_current_idx].id2 = right_elem->id;
			*merged_y_current_idx += 1;
		}
	}
}

__device__
void d_merge_blocks_vertically(int di_lb_bottom_left_y, int step, int channels,
		int ai_x, int width, int ai_y, int merged_x_start_idx,
		int *merged_x_current_idx, uchar* data, element* elements,
		elements_pair* merged_x) {

	int block_width = width / gridDim.x; // TODO

	for (int i = 0; i < 2*block_width; i++) {
		int di_tlb_bottom = di_lb_bottom_left_y + i * channels;
		int di_blb_top = di_tlb_bottom + step;
		int ai_tb = ai_x + i + width * ai_y;
		int ai_bb = ai_tb + width;
		if (!d_is_empty(data, di_tlb_bottom) && !d_is_empty(data, di_blb_top)) {
			element* top_elem = &(elements[ai_tb]);
			element* bottom_elem = &(elements[ai_bb]);
			if (d_already_merged(merged_x, merged_x_start_idx, *merged_x_current_idx, top_elem, bottom_elem))
				continue;

			d_merge_elements(elements, top_elem, bottom_elem, width);
			merged_x[*merged_x_current_idx].id1 = top_elem->id;
			merged_x[*merged_x_current_idx].id2 = bottom_elem->id;
			*merged_x_current_idx += 1;
		}
	}
}

__device__
void d_merge_elements(element* elements, element* e1, element* e2, int width) {
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

__device__
bool d_is_empty(uchar* data, int addr) {
	return data[addr+1] == 0 && data[addr+2] == 0; // TODO channels?
}

__device__
bool d_already_merged(elements_pair* merged, int merged_start_idx, int merged_last_idx,
		element* e1, element* e2) {
	for(int i = merged_start_idx; i < merged_last_idx; i++) {
		if(merged[i].id1 == e1->id && merged[i].id2 == e2->id)
			return true;
	}

	return false;
}

}}
