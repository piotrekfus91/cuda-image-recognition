#include "cir/gpuprocessing/GpuMomentCounter.h"
#include "cir/gpuprocessing/count_moments.cuh"
#include "cir/common/config.h"
#include "cir/common/concurrency/StreamHandler.h"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cir::common;
using namespace cir::common::concurrency;

namespace cir { namespace gpuprocessing {

GpuMomentCounter::GpuMomentCounter() {

}

GpuMomentCounter::~GpuMomentCounter() {

}

void GpuMomentCounter::init(int width, int height) {

}

double* GpuMomentCounter::countHuMoments(MatWrapper& matWrapper) {
	cv::gpu::GpuMat mat = matWrapper.getGpuMat();
	uchar* data = mat.data;
	int width = mat.cols;
	int height = mat.rows;
	int step = mat.step;

	double* rawMoments = new double[10];

	count_raw_moments(data, width, height, step, rawMoments, StreamHandler::nativeStream());

	double M00 = rawMoments[0];
	double M01 = rawMoments[1];
	double M10 = rawMoments[2];
	double M11 = rawMoments[3];
	double M02 = rawMoments[4];
	double M20 = rawMoments[5];
	double M21 = rawMoments[6];
	double M12 = rawMoments[7];
	double M30 = rawMoments[8];
	double M03 = rawMoments[9];

	delete rawMoments;

	cv::Moments cvMoments(M00, M10, M01, M20, M11, M02, M30, M21, M12, M03);

	double* huMoments = (double*) malloc(sizeof(double) * HU_MOMENTS_NUMBER);

	cv::HuMoments(cvMoments, huMoments);

	return huMoments;
}

}}
