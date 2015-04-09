#include <vector_types.h>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include "cir/gpuprocessing/region_splitting_segmentate.cuh"
#include "cir/gpuprocessing/segmentate_base.cuh"
#include "cir/common/cuda_host_util.cuh"
#include "cir/common/config.h"

#define CHANNELS 3
#define THREADS 16

using namespace cir::common;
using namespace cir::common::logger;

namespace cir { namespace gpuprocessing {

void region_splitting_segmentate_init(int width, int height) {

}

SegmentArray* region_splitting_segmentate(uchar* data, int step, int channels, int width, int height,
		cudaStream_t stream) {
	int sumBlocksNumber = (width*height+THREADS*THREADS-1)/(THREADS*THREADS);

	element* elements;
	Segment* segments;
	int* partialSums;

	HANDLE_CUDA_ERROR(cudaHostAlloc((void**) &elements, sizeof(element) * width * height, cudaHostAllocDefault));
	HANDLE_CUDA_ERROR(cudaHostAlloc((void**) &segments, sizeof(Segment) * width * height, cudaHostAllocDefault));
	HANDLE_CUDA_ERROR(cudaHostAlloc((void**) &partialSums, sizeof(int) * sumBlocksNumber, cudaHostAllocDefault));

	element* d_elements;
	Segment* d_segments;
	int* d_partialSums;

	HANDLE_CUDA_ERROR(cudaMalloc((void**) &d_elements, sizeof(element) * width * height));
	HANDLE_CUDA_ERROR(cudaMalloc((void**) &d_segments, sizeof(Segment) * width * height));
	HANDLE_CUDA_ERROR(cudaMalloc((void**) &d_partialSums, sizeof(int) * sumBlocksNumber));

	for(int i = 0; i < width * height; i++) {
		elements[i].id = i;
		elements[i].valid = true;

		segments[i] = createSimpleSegment(i % width, i / width);
	}

	HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_elements, elements, sizeof(element) * width * height,
			cudaMemcpyHostToDevice, stream));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_segments, segments, sizeof(Segment) * width * height,
			cudaMemcpyHostToDevice, stream));

	dim3 blocks((width+THREADS-1)/THREADS, (height+THREADS-1)/THREADS);
	dim3 threads(THREADS, THREADS);

	k_remove_empty_segments<<<blocks, threads, 0, stream>>>(data, width, height, step, d_elements);
	HANDLE_CUDA_ERROR(cudaGetLastError());

	int greaterDim = width > height ? width : height;

	for(int i = 1; i < greaterDim; i = 2 * i) {
		int block_width = i;
		int block_height = i;

		dim3 blocks((width+i*THREADS-1)/(i*THREADS), (height+i*THREADS-1)/(i*THREADS));
		dim3 threads(THREADS, THREADS);

//		KERNEL_MEASURE_START(stream)

		k_region_splitting_segmentate<<<blocks, threads, 0, stream>>>(data, d_elements, d_segments, step,
				channels, width, height, block_width, block_height);
		HANDLE_CUDA_ERROR(cudaGetLastError());

//		KERNEL_MEASURE_END("Segmentate", stream)

//		HANDLE_CUDA_ERROR(cudaMemcpy(elements, d_elements, sizeof(element) * width * height, cudaMemcpyDeviceToHost));
//		for(int x = 0; x < width; x++) {
//			for(int y = 0; y < height; y++) {
//				std::cout << std::setw(6) << elements[x*height + y].id << " ";
//			}
//
//			std::cout << std::endl;
//		}
//		std::cout << "-----------" << std::endl;
//		HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
	}

	HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

	dim3 blocksForSum(sumBlocksNumber);
	dim3 threadsForSum(THREADS*THREADS);

//	KERNEL_MEASURE_START(stream)
	k_count_applicable_segments<<<blocksForSum, threadsForSum, 0, stream>>>(d_elements, d_segments, width*height, _min_size, d_partialSums);
	HANDLE_CUDA_ERROR(cudaGetLastError());
//	KERNEL_MEASURE_END("Segmentate sum", stream)

	HANDLE_CUDA_ERROR(cudaMemcpyAsync(elements, d_elements, sizeof(element) * width * height,
			cudaMemcpyDeviceToHost, stream));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(segments, d_segments, sizeof(Segment) * width * height,
			cudaMemcpyDeviceToHost, stream));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(partialSums, d_partialSums, sizeof(int) * sumBlocksNumber,
			cudaMemcpyDeviceToHost, stream));

	HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

	int foundSegmentsSize = 0;
	for(int j = 0; j < sumBlocksNumber; j++) {
		foundSegmentsSize += partialSums[j];
	}

	SegmentArray* segmentArray = (SegmentArray*) malloc(sizeof(SegmentArray));
	segmentArray->size = foundSegmentsSize;

	if(foundSegmentsSize > 0) {
		Segment** appliedSegments = (Segment**) malloc(sizeof(Segment*) * foundSegmentsSize);
		int currentSegmentIndex = 0;
		for(int j = 0; j < width*height; j++) {
			if(elements[j].valid) {
				bool segment_applicable = false;
				d_is_segment_applicable_rs(&(segments[j]), &segment_applicable, _min_size);
				if(segment_applicable) {
					Segment segment = segments[j];
					appliedSegments[currentSegmentIndex++] = copySegment(&segment);
				}
			}
		}
		segmentArray->segments = appliedSegments;
	} else {
		segmentArray->segments = NULL;
	}

	HANDLE_CUDA_ERROR(cudaFree(d_elements));
	HANDLE_CUDA_ERROR(cudaFree(d_segments));
	HANDLE_CUDA_ERROR(cudaFree(d_partialSums));

	HANDLE_CUDA_ERROR(cudaFreeHost(elements));
	HANDLE_CUDA_ERROR(cudaFreeHost(segments));
	HANDLE_CUDA_ERROR(cudaFreeHost(partialSums));

	return segmentArray;
}

void region_splitting_segmentate_shutdown() {

}

__global__
void k_remove_empty_segments(uchar* data, int width, int height, int step, element* elements) {
	int ai_x = blockDim.x * blockIdx.x + threadIdx.x;
	if(ai_x >= width)
		return;

	int ai_y = blockDim.y * blockIdx.y + threadIdx.y;
	if(ai_y >= height)
		return;

	int di = ai_x * CHANNELS + ai_y * step;
	uchar saturation = data[di+1];
	uchar value = data[di+2];

	if(saturation == 0 && value == 0) {
		element* elem = &(elements[ai_x + width * ai_y]);
		elem->id = -1;
		elem->valid = false;
	}
}

__global__
void k_count_applicable_segments(element* elements, Segment* segments,
		int total_size, int min_size, int* partialSums) {
	__shared__ int cache[THREADS*THREADS];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid > total_size)
		return;

	bool is_segment_applicable = false;
	d_is_segment_applicable_rs(&(segments[tid]), &is_segment_applicable, min_size);

	element* elem = &(elements[tid]);

	if(is_segment_applicable && elem->valid) {
		cache[threadIdx.x] = 1;
	} else {
		cache[threadIdx.x] = 0;
	}
	__syncthreads();

	for(int i = blockDim.x / 2; i > 0; i /= 2) {
		if(threadIdx.x < i && tid + i < total_size) {
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0)
		partialSums[blockIdx.x] = cache[0];
}

// tlb - top left block
// brb - bottom right block
// di_ - data index
// ai - array index
__global__
void k_region_splitting_segmentate(uchar* data, element* elements, Segment* segments,
		int step, int channels, int width, int height, int block_width, int block_height) {
	int ai_x = blockDim.x * blockIdx.x + threadIdx.x;
	int ai_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(ai_x % 2 != 0 || ai_y % 2 != 0)
		return;

	ai_x = ai_x * block_width;
	ai_y = ai_y * block_height;

	if(ai_x >= width || ai_y >= height)
		return;

	// top left and top right
	int di_tlb_top_right_x = (ai_x + block_width - 1) * channels + ai_y * step;
	int ai_lb_top_right_x = ai_x + block_width - 1;

	d_merge_blocks_horizontally(di_tlb_top_right_x, step, channels, ai_lb_top_right_x, width, height,
			ai_y, data, elements, segments, block_width, block_height);

	// bottom left and bottom right
	int di_blb_top_right_x = di_tlb_top_right_x + block_height * step;
	int blb_ai_y = ai_y + block_height;

	d_merge_blocks_horizontally(di_blb_top_right_x, step, channels, ai_lb_top_right_x, width, height,
			blb_ai_y, data, elements, segments, block_width, block_height);

	// top left/right and bottom left/right
	int di_tb_bottom_left_y = ai_x * channels + (ai_y + block_height - 1) * step;
	d_merge_blocks_vertically(di_tb_bottom_left_y, step, channels, ai_x, width, height, ai_y + block_height - 1,
			data, elements, segments, block_width, block_height);
}

__device__
void d_merge_blocks_horizontally(int di_lb_top_right_x, int step,
		int channels, int ai_x, int width, int height, int ai_y, uchar* data, element* elements,
		Segment* segments, int block_width, int block_height) {

	for (int i = 0; i < block_height; i++) {
		int di_tlb_right = di_lb_top_right_x + i * step;
		int di_trb_left = di_tlb_right + channels;
		int ai_tlb = ai_x + width * (i + ai_y);
		int ai_trb = ai_tlb + 1;

		if(ai_trb % width < ai_tlb % width || ai_trb > width * height)
			return;

		if (!d_is_empty(data, di_tlb_right) && !d_is_empty(data, di_trb_left)) {
			element* left_elem = &(elements[ai_tlb]);
			element* right_elem = &(elements[ai_trb]);

			int left_elem_id = left_elem->id;
			int right_elem_id = right_elem->id;

			for(int j = 0; j < block_height; j++) {
				int ai_tlb_right = ai_x + width * j + ai_y * width;
				d_try_merge(ai_tlb_right, right_elem_id, left_elem_id, width, height,
						elements, segments, false);

				int ai_trb_left = ai_tlb_right + 1;
				d_try_merge(ai_trb_left, right_elem_id, left_elem_id, width, height,
						elements, segments);

				int x_trb_left = ai_trb_left % width;
				int normalized_width = x_trb_left + block_width - 1 > width ? width - x_trb_left + 1 : block_width;

				int ai_trb_right = ai_trb_left + normalized_width - 1;
				d_try_merge(ai_trb_right, right_elem_id, left_elem_id, width, height,
						elements, segments);

				int ai_tlb_left = ai_trb_left - block_width;
				d_try_merge(ai_tlb_left, right_elem_id, left_elem_id, width, height,
						elements, segments, false);
			}

			for(int j = 0; j < block_width; j++) {
				int ai_trb_top = ai_x + j + 1 + ai_y * width;
				d_try_merge(ai_trb_top, right_elem_id, left_elem_id, width, height,
						elements, segments);

				int ai_tlb_top = ai_trb_top - block_width;
				d_try_merge(ai_tlb_top, right_elem_id, left_elem_id, width, height,
						elements, segments, false);

				int y_trb_top = ai_trb_top / width;
				int normalized_height = y_trb_top + block_height - 1 > height ? height - y_trb_top + 1 : block_height;

				int ai_trb_bottom = (normalized_height - 1) * width + ai_trb_top;
				d_try_merge(ai_trb_bottom, right_elem_id, left_elem_id, width, height,
						elements, segments);

				int ai_tlb_bottom = ai_trb_bottom - block_width;
				d_try_merge(ai_tlb_bottom, right_elem_id, left_elem_id, width, height,
						elements, segments, false);
			}
		}
	}
}

__device__
void d_merge_blocks_vertically(int di_lb_bottom_left_y, int step,
		int channels, int ai_x, int width, int height, int ai_y, uchar* data, element* elements,
		Segment* segments, int block_width, int block_height) {

	for (int i = 0; i < 2*block_width; i++) {
		int di_tlb_bottom = di_lb_bottom_left_y + i * channels;
		int di_blb_top = di_tlb_bottom + step;
		int ai_tb = ai_x + i + width * ai_y;
		int ai_bb = ai_tb + width;

		if(ai_bb / width > height || ai_bb > width * height)
			return;

		if (!d_is_empty(data, di_tlb_bottom) && !d_is_empty(data, di_blb_top)) {
			element* top_elem = &(elements[ai_tb]);
			element* bottom_elem = &(elements[ai_bb]);

			int top_elem_id = top_elem->id;
			int bottom_elem_id = bottom_elem->id;

			for(int j = 0; j < 2*block_width; j++) {
				int ai_bb_top = ai_x + width + j + ai_y * width;
				d_try_merge(ai_bb_top, bottom_elem_id, top_elem_id, width, height,
						elements, segments);

				int ai_tb_top = ai_bb_top - block_height * width;
				d_try_merge(ai_tb_top, bottom_elem_id, top_elem_id, width, height,
						elements, segments, false);

				int y_bb_top = ai_bb_top / width;
				int normalized_height = y_bb_top + block_height - 1 > height ? height - y_bb_top + 1 : block_height;

				int ai_bb_bottom = ai_bb_top + (normalized_height-1) * width;
				d_try_merge(ai_bb_bottom, bottom_elem_id, top_elem_id, width, height,
						elements, segments);

				int ai_tb_bottom = ai_bb_bottom - block_height * width;
				d_try_merge(ai_tb_bottom, bottom_elem_id, top_elem_id, width, height,
						elements, segments, false);
			}

			for(int j = 0; j < block_height; j++) {
				int ai_bb_left = ai_x + (j+1) * width + ai_y * width;
				d_try_merge(ai_bb_left, bottom_elem_id, top_elem_id, width, height,
						elements, segments);

				int ai_tb_left = ai_bb_left - block_height * width;
				d_try_merge(ai_tb_left, bottom_elem_id, top_elem_id, width, height,
						elements, segments, false);

				int x_bb_left = ai_bb_left % width;
				int normalized_width = x_bb_left + 2*block_width - 1 > width ? width - x_bb_left + 1 : 2 * block_width;

				int ai_bb_right = ai_bb_left + normalized_width - 1;
				d_try_merge(ai_bb_right, bottom_elem_id, top_elem_id, width, height,
						elements, segments);

				int ai_tb_right = ai_bb_right - block_height * width;
				d_try_merge(ai_tb_right, bottom_elem_id, top_elem_id, width, height,
						elements, segments, false);
			}
		}
	}
}

__device__
void d_try_merge(int idx, int current_elem_id, int id_to_set, int width, int height,
		element* elements, Segment* segments, bool invalidate_all) {
	if(idx < width * height) {
		element* elem = &(elements[idx]);
		if(elem->id == current_elem_id) {
			Segment* segm1 = &(segments[id_to_set]);
			Segment* segm2 = &(segments[elem->id]);
			if(id_to_set != elem->id) {
				element* elemToInvalidate = &(elements[elem->id]);
				if(invalidate_all)
					elemToInvalidate->valid = false;
				elemToInvalidate->id = id_to_set;
			}
			d_merge_segments_rs(segm1, segm2);
			if(invalidate_all)
				elem->valid = false;
			elem->id = id_to_set;
		}
	}
}

__device__
bool d_is_empty(uchar* data, int addr) {
	return data[addr+1] == 0 && data[addr+2] == 0;
}

__device__ __host__
void d_is_segment_applicable_rs(Segment* segment, bool* is_applicable, int min_size) {
	int width = abs(segment->rightX - segment->leftX);
	int height = abs(segment->topY - segment->bottomY);
	*is_applicable = width >= min_size && height >= min_size;
}

__device__
void d_merge_segments_rs(Segment* segm1, Segment* segm2) {
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
