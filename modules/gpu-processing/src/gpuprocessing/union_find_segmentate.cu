#include "cir/gpuprocessing/segmentate_base.cuh"
#include "cir/gpuprocessing/union_find_segmentate.cuh"
#include "cir/common/cuda_host_util.cuh"
#include <iostream>
#include <iomanip>
#include <map>

#define THREADS 16

using namespace cir::common;

namespace cir { namespace gpuprocessing {

int* ids;
int* d_ids;
bool* changed;
bool* d_changed;

void union_find_segmentate_init(int width, int height) {
	segments = (Segment*) malloc(sizeof(Segment) * width * height);
	ids = (int*) malloc(sizeof(int) * width * height);
	changed = (bool*) malloc(sizeof(bool));

	HANDLE_CUDA_ERROR(cudaMalloc(&d_segments, sizeof(Segment) * width * height));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_ids, sizeof(int) * width * height));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_changed, sizeof(bool)));
}

void union_find_segmentate_shutdown() {
	free(segments);
	free(ids);
	free(changed);

	HANDLE_CUDA_ERROR(cudaFree(d_segments));
	HANDLE_CUDA_ERROR(cudaFree(d_ids));
	HANDLE_CUDA_ERROR(cudaFree(d_changed));
}

SegmentArray* union_find_segmentate(uchar* data, int step, int channels, int width, int height) {
	dim3 threads(THREADS, THREADS);
	dim3 blocks((width+THREADS-1)/THREADS, (height+THREADS-1)/THREADS);

	k_init_internal_structures<<<blocks, threads>>>(data, width, height, step, channels, d_segments, d_ids);
	HANDLE_CUDA_ERROR(cudaGetLastError());

	*changed = true;
	while(*changed) {
		*changed = false;
		HANDLE_CUDA_ERROR(cudaMemcpy(d_changed, changed, sizeof(bool), cudaMemcpyHostToDevice));

		k_prepare_best_neighbour<<<blocks, threads>>>(d_ids, d_segments, width, height, d_changed);
		HANDLE_CUDA_ERROR(cudaGetLastError());

		k_find_best_root<<<blocks, threads>>>(d_ids, width, height);
		HANDLE_CUDA_ERROR(cudaGetLastError());

		HANDLE_CUDA_ERROR(cudaMemcpy(changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
	}

	HANDLE_CUDA_ERROR(cudaMemcpy(ids, d_ids, sizeof(int)*width*height, cudaMemcpyDeviceToHost));
	HANDLE_CUDA_ERROR(cudaMemcpy(segments, d_segments, sizeof(Segment)*width*height, cudaMemcpyDeviceToHost));

	std::map<int, Segment*> appliedSegments;
	int total = 0;
	for(int i = 0; i < width*height; i++) {
		if(i == ids[i]) {
			Segment* segm = &segments[i];
			bool segm_applied = false;
			d_is_segment_applicable(segm, &segm_applied, _min_size);
			if(segm_applied) {
				total++;
				appliedSegments[i] = segm;
			}
		}
	}

	SegmentArray* segmentArray = (SegmentArray*) malloc(sizeof(SegmentArray));

	if(total > 0) {
		for(int i = 0; i < width * height; i++) {
			int currentSegmentId = ids[i];
			if(currentSegmentId == -1)
				continue;

			Segment* currentSegment = &segments[i];
			Segment* targetSegment = appliedSegments[currentSegmentId];
			d_merge_segments(currentSegment, targetSegment);
		}

		int idx = 0;
		Segment** segmentsToSet = (Segment**) malloc(sizeof(Segment*) * total);
		for(std::map<int, Segment*>::iterator it = appliedSegments.begin(); it != appliedSegments.end(); it++) {
			Segment* segm = (*it).second;
			Segment* copy = copySegment(segm);
			segmentsToSet[idx] = copy;
			idx++;
		}
		segmentArray->segments = segmentsToSet;
		segmentArray->size = total;
	} else {
		segmentArray->size = 0;
		segmentArray->segments = NULL;
	}

	return segmentArray;
}

__global__
void k_prepare_best_neighbour(int* ids, Segment* segments, int width, int height, bool* changed) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	if(x >= width)
		return;

	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if(y >= height)
		return;

	int pos = d_count_pos(x, y, width, height);
	if(ids[pos] == -1)
		return;

	if(x > 0) {
		int neighbourPos = pos - 1;
		d_unite(pos, neighbourPos, ids, segments, changed);
	}

	if(x < width - 1) {
		int neighbourPos = pos + 1;
		d_unite(pos, neighbourPos, ids, segments, changed);
	}

	if(y > 0) {
		int neighbourPos = pos - width;
		d_unite(pos, neighbourPos, ids, segments, changed);
	}

	if(y < height - 1) {
		int neighbourPos = pos + width;
		d_unite(pos, neighbourPos, ids, segments, changed);
	}
}

__global__
void k_find_best_root(int* ids, int width, int height) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	if(x >= width)
		return;

	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if(y >= height)
		return;

	int pos = d_count_pos(x, y, width, height);
	int currentId = ids[pos];
	if(currentId != -1 && currentId != pos) {
		ids[pos] = d_find_root(ids, pos);
	}
}

__device__
int d_find_root(int* ids, int pos) {
	while(ids[pos] != pos) {
		pos = ids[pos];
	}
	return pos;
}

__device__
void d_unite(int pos1, int pos2, int* ids, Segment* segments, bool* changed) {
	int id1 = ids[pos1];
	if(id1 == -1)
		return;

	int id2 = ids[pos2];
	if(id2 == -1)
		return;

	int root1 = d_find_root(ids, pos1);
	int root2 = d_find_root(ids, pos2);

	if(root1 < root2) {
		d_merge_segments(&segments[ids[root1]], &segments[ids[root2]]);
		ids[root2] = root1;
		*changed = true;
	} else if(root1 > root2) {
		d_merge_segments(&segments[ids[root1]], &segments[ids[root2]]);
		ids[root1] = root2;
		*changed = true;
	}
}

__device__ __host__
int d_count_pos(int x, int y, int width, int height) {
	return x + width * y;
}

__global__
void k_init_internal_structures(uchar* data, int width, int height, int step, int channels,
		Segment* segments, int* ids) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if(x >= width)
		return;

	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(y >= height)
		return;

	int ai = x + width * y;
	int di = x * channels + y * step;

	int* id = &ids[ai];
	if(channels == 1) {
		uchar val = data[di];
		if(val == 0) {
			*id = -1;
		} else {
			*id = ai;
		}
	} else if(channels == 3) {
		uchar saturation = data[di+1];
		uchar value = data[di+2];

		if(saturation == 0 && value == 0) {
			*id = -1;
		} else {
			*id = ai;
		}
	}

	Segment* segm = &segments[ai];
	segm->leftX = x;
	segm->rightX = x;
	segm->topY = y;
	segm->bottomY = y;
}

__device__ __host__
void d_merge_segments(Segment* segm1, Segment* segm2) {
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

__device__ __host__
void d_is_segment_applicable(Segment* segment, bool* is_applicable, int min_size) {
	int width = abs(segment->rightX - segment->leftX);
	int height = abs(segment->topY - segment->bottomY);
	*is_applicable = width >= min_size && height >= min_size;
}

}}
