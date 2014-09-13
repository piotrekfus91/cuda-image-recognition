#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "cir/common/MatWrapper.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/gpuprocessing/region_splitting_segmentate.cuh"
#include "cir/common/test_file_loader.h"

using namespace cir::common;
using namespace cir::gpuprocessing;

class region_splitting_segmentate : public ::testing::Test {

};

TEST_F(region_splitting_segmentate, 8x8) {
	cv::Mat mat = cv::imread(getTestFile("gpu-processing", "8x8.bmp"));
	cv::gpu::GpuMat gpuMat;
	gpuMat.upload(mat);
	MatWrapper img(gpuMat);

	GpuImageProcessingService service;
	img = service.bgrToHsv(img);
}
