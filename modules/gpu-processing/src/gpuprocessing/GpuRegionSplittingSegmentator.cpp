#include "cir/gpuprocessing/GpuRegionSplittingSegmentator.h"
#include "cir/gpuprocessing/region_splitting_segmentate.cuh"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuRegionSplittingSegmentator::GpuRegionSplittingSegmentator() {

}

GpuRegionSplittingSegmentator::~GpuRegionSplittingSegmentator() {

}

SegmentArray* GpuRegionSplittingSegmentator::segmentate(const MatWrapper& matWrapper) {
	cv::gpu::GpuMat mat = matWrapper.getGpuMat();
	SegmentArray* segmentArray = (SegmentArray*)malloc(sizeof(SegmentArray));
	region_splitting_segmentate(mat.data, mat.step, mat.channels(), mat.cols, mat.rows);
	return segmentArray;
}

}}
