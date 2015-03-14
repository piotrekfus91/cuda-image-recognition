#include "cir/gpuprocessing/GpuUnionFindSegmentator.h"
#include "cir/gpuprocessing/union_find_segmentate.cuh"
#include "cir/gpuprocessing/segmentate_base.cuh"
#include "cir/common/concurrency/StreamHandler.h"

using namespace cir::common;
using namespace cir::common::concurrency;

namespace cir { namespace gpuprocessing {

GpuUnionFindSegmentator::GpuUnionFindSegmentator() {

}

GpuUnionFindSegmentator::~GpuUnionFindSegmentator() {

}

void GpuUnionFindSegmentator::init(int width, int height) {

}

void GpuUnionFindSegmentator::setMinSize(int size) {
	set_segment_min_size(size);
}

SegmentArray* GpuUnionFindSegmentator::segmentate(const MatWrapper& matWrapper) {
	cv::gpu::GpuMat mat = matWrapper.getGpuMat();
	SegmentArray* segmentArray = union_find_segmentate(mat.data, mat.step, mat.channels(),
			mat.cols, mat.rows, StreamHandler::nativeStream());
	return segmentArray;
}

}}
