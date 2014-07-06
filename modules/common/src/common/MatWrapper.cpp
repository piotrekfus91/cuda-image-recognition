#include "common/MatWrapper.h"
#include "common/exception/InvalidMatTypeException.h"

using namespace cv;
using namespace cv::gpu;
using namespace cir::common;
using namespace cir::common::exception;

MatWrapper::MatWrapper(const Mat& mat) : _mat(mat), _matType(MAT) {

}

MatWrapper::MatWrapper(const GpuMat& gpuMat) : _gpuMat(gpuMat), _matType(GPU_MAT) {

}

Mat MatWrapper::getMat() {
	if(_matType != MAT)
		throw InvalidMatTypeException();
	return _mat;
}

GpuMat MatWrapper::getGpuMat() {
	if(_matType != GPU_MAT)
		throw InvalidMatTypeException();
	return _gpuMat;
}

MatWrapper::MAT_TYPE MatWrapper::getType() {
	return _matType;
}
