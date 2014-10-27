#include "cir/common/MatWrapper.h"
#include "cir/common/exception/InvalidMatTypeException.h"
#include "cir/common/exception/UnsupportedDataTypeException.h"

using namespace cv;
using namespace cv::gpu;
using namespace cir::common;
using namespace cir::common::exception;

MatWrapper::MatWrapper(const Mat& mat) : _mat(mat), _matType(MAT), _colorScheme(BGR) {
	validateType();
}

MatWrapper::MatWrapper(const GpuMat& gpuMat) : _gpuMat(gpuMat), _matType(GPU_MAT), _colorScheme(BGR) {
	validateType();
}

MatWrapper MatWrapper::clone() const {
	if(_matType == MAT) {
		cv::Mat mat(_mat.clone());
		MatWrapper mw(mat);
		mw.setColorScheme(_colorScheme);
		return mw;
	} else if(_matType == GPU_MAT) {
		cv::gpu::GpuMat gpuMat(_gpuMat.clone());
		MatWrapper mw(gpuMat);
		mw.setColorScheme(_colorScheme);
		return mw;
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

int MatWrapper::getWidth() const {
	if(_matType == MAT)
		return _mat.cols;
	else if(_matType == GPU_MAT)
		return _gpuMat.cols;

	return 0;
}

int MatWrapper::getHeight() const {
	if(_matType == MAT)
		return _mat.rows;
	else if(_matType == GPU_MAT)
		return _gpuMat.cols;

	return 0;
}

void MatWrapper::validateType() {
	int type = -1;
	if(_matType == MAT)
		type = _mat.type();
	else if(_matType == GPU_MAT)
		type = _gpuMat.type();

	int dataType = type % 8;
	if(dataType != 0)
		throw new UnsupportedDataTypeException();
}
