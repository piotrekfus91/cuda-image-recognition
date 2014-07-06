#include "cir/common/MatWrapper.h"
#include "cir/common/exception/InvalidMatTypeException.h"

using namespace cv;
using namespace cv::gpu;
using namespace cir::common;
using namespace cir::common::exception;

MatWrapper::MatWrapper(const Mat& mat) : _mat(mat), _matType(MAT) {

}

MatWrapper::MatWrapper(const GpuMat& gpuMat) : _gpuMat(gpuMat), _matType(GPU_MAT) {

}

Mat MatWrapper::getMat() const {
	if(_matType != MAT)
		throw InvalidMatTypeException();
	return _mat;
}

GpuMat MatWrapper::getGpuMat() const {
	if(_matType != GPU_MAT)
		throw InvalidMatTypeException();
	return _gpuMat;
}

MatWrapper::MAT_TYPE MatWrapper::getType() const {
	return _matType;
}
