#include "cir/gpuprocessing/GpuRegionSplittingSegmentator.h"
#include "cir/gpuprocessing/region_splitting_segmentate.cuh"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuRegionSplittingSegmentator::GpuRegionSplittingSegmentator() {

}

GpuRegionSplittingSegmentator::~GpuRegionSplittingSegmentator() {
	region_splitting_segmentate_shutdown();
}

void GpuRegionSplittingSegmentator::init(int width, int height) {
	region_splitting_segmentate_init(width, height);
}

void GpuRegionSplittingSegmentator::setMinSize(int size) {
	set_min_segment_size(size);
}

SegmentArray* GpuRegionSplittingSegmentator::segmentate(const MatWrapper& matWrapper) {
	cv::gpu::GpuMat mat = matWrapper.getGpuMat();
	SegmentArray* segmentArray = region_splitting_segmentate(mat.data, mat.step, mat.channels(), mat.cols, mat.rows);
	return segmentArray;
}

}}
