#include "cir/common/MatWrapper.h"
#include "cir/common/exception/InvalidMatTypeException.h"

using namespace cv;
using namespace cv::gpu;
using namespace cir::common;
using namespace cir::common::exception;

MatWrapper::MatWrapper(const Mat& mat) : _mat(mat), _matType(MAT), _colorScheme(BGR) {

}

MatWrapper::MatWrapper(const GpuMat& gpuMat) : _gpuMat(gpuMat), _matType(GPU_MAT), _colorScheme(BGR) {

}

MatWrapper MatWrapper::clone() const {
	if(_matType == MAT) {
		cv::Mat mat(_mat.clone());
		return MatWrapper(mat);
	} else if(_matType == GPU_MAT) {
		cv::gpu::GpuMat gpuMat(_gpuMat.clone());
		return MatWrapper(gpuMat);
	} else {
		throw InvalidMatTypeException();
	}
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

MatWrapper::COLOR_SCHEME MatWrapper::getColorScheme() const {
	return _colorScheme;
}

void MatWrapper::setColorScheme(const COLOR_SCHEME colorScheme) {
	_colorScheme = colorScheme;
}
