#include <iostream>
#include "cir/gpuprocessing/GpuUnionFindSegmentator.h"
#include "cir/gpuprocessing/union_find_segmentate.cuh"
#include "cir/gpuprocessing/segmentate_base.cuh"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuUnionFindSegmentator::GpuUnionFindSegmentator() {

}

GpuUnionFindSegmentator::~GpuUnionFindSegmentator() {
	union_find_segmentate_shutdown();
}

void GpuUnionFindSegmentator::init(int width, int height) {
	union_find_segmentate_init(width, height);
}

void GpuUnionFindSegmentator::setMinSize(int size) {
	set_segment_min_size(size);
}

SegmentArray* GpuUnionFindSegmentator::segmentate(const MatWrapper& matWrapper) {
	cv::gpu::GpuMat mat = matWrapper.getGpuMat();
	SegmentArray* segmentArray = union_find_segmentate(mat.data, mat.step, mat.channels(), mat.cols, mat.rows);
	return segmentArray;
}

}}
