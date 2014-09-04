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
	}

	int* merged_y = (int*) malloc(sizeof(int) * width * height);
	for(int i = 0; i < width*height; i++) merged_y[i] = -1;
	int* merged_x = (int*) malloc(sizeof(int) * width * height);

	element* d_elements;
	int* d_merged_y;
	int* d_merged_x;

	cudaMalloc((void**) &d_elements, sizeof(element) * width * height);
	cudaMalloc((void**) &d_merged_y, sizeof(int) * width * height);
	cudaMalloc((void**) &d_merged_x, sizeof(int) * width * height);

	cudaMemcpy(d_elements, elements, sizeof(element) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_merged_y, merged_y, sizeof(int) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_merged_x, merged_x, sizeof(int) * width * height, cudaMemcpyHostToDevice);

	int greaterDim = width > height ? width : height;

	for(int i = 1; i < greaterDim; i = 2 * i) {
		dim3 blocks((width+i-1)/i, (height+i-1)/i);
		dim3 threads(1, 1);
		k_region_splitting_segmentate<<<blocks, threads>>>(data, d_merged_y, d_merged_x, d_elements,
				step, channels, width, height);
		cudaMemcpy(elements, d_elements, sizeof(element) * width * height, cudaMemcpyDeviceToHost);
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
void k_region_splitting_segmentate(uchar* data, int* merged_y, int* merged_x, element* elements,
		int step, int channels, int width, int height) {
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

	int di_tlb_top_right_x = (ai_x + block_width - 1) * channels + ai_y * step;
	int ai_lb_top_right_x = ai_x + block_width - 1;

	merge_blocks_horizontally(di_tlb_top_right_x, step, channels, ai_lb_top_right_x, width,
			ai_y, merged_y_start_idx, &merged_y_current_idx, data, elements,
			merged_y);

	int di_blb_top_right_x = di_tlb_top_right_x + block_height * step;
	int blb_ai_y = ai_y + block_height;

	merge_blocks_horizontally(di_blb_top_right_x, step, channels, ai_lb_top_right_x, width,
			blb_ai_y, merged_y_start_idx, &merged_y_current_idx, data, elements,
			merged_y);
}

__device__
void merge_blocks_horizontally(int di_lb_top_right_x, int step, int channels,
		int ai_x, int width, int ai_y, int merged_y_start_idx,
		int *merged_y_current_idx, uchar* data, element* elements,
		int* merged_y) {

	int block_height = width / gridDim.x; // TODO

	for (int i = 0; i < block_height; i++) {
		int di_tlb_right = di_lb_top_right_x + i * step;
		int di_trb_left = di_tlb_right + channels;
		int ai_tlb = ai_x + width * (i + ai_y);
		int ai_trb = ai_tlb + 1;
		if (!d_is_empty(data, di_tlb_right) && !d_is_empty(data, di_trb_left)) {
			element* left_elem = &(elements[ai_tlb]);
			element* right_elem = &(elements[ai_trb]);
			if (d_already_merged(merged_y, merged_y_start_idx, *merged_y_current_idx, left_elem))
				continue;

			d_merge_elements(elements, left_elem, right_elem, width);
			merged_y[*merged_y_current_idx] = left_elem->id;
			*merged_y_current_idx += 1;
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
}

__device__
bool d_is_empty(uchar* data, int addr) {
	return data[addr+1] == 0 && data[addr+2] == 0; // TODO
}

__device__
bool d_already_merged(int* merged, int merged_start_idx, int merged_last_idx, element* elem) {
	for(int i = merged_start_idx; i < merged_last_idx; i++) {
		if(merged[i] == elem->id)
			return true;
	}

	return false;
}

}}
