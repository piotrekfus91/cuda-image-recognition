#include <gtest/gtest.h>
#include "cir/common/MatWrapper.h"
#include "cir/common/exception/InvalidMatTypeException.h"

using namespace cv;
using namespace cv::gpu;
using namespace cir::common;
using namespace cir::common::exception;

class MatWrapperTest : public ::testing::Test {

};

TEST_F(MatWrapperTest, SettingMat) {
	Mat mat;
	MatWrapper matWrapper(mat);
	MatWrapper::MAT_TYPE matType = MatWrapper::MAT;
	ASSERT_EQ(matType, matWrapper.getType());
}

TEST_F(MatWrapperTest, GettingGpuMatInsteadOfMat) {
	Mat mat;
	MatWrapper matWrapper(mat);
	ASSERT_THROW(matWrapper.getGpuMat(), InvalidMatTypeException);
}

TEST_F(MatWrapperTest, SettingGpuMat) {
	GpuMat gpuMat;
	MatWrapper matWrapper(gpuMat);
	MatWrapper::MAT_TYPE gpuMatType = MatWrapper::GPU_MAT;
	ASSERT_EQ(gpuMatType, matWrapper.getType());
}

TEST_F(MatWrapperTest, GettingMatInsteadOfGpuMat) {
	GpuMat gpuMat;
	MatWrapper matWrapper(gpuMat);
	ASSERT_THROW(matWrapper.getMat(), InvalidMatTypeException);
}
