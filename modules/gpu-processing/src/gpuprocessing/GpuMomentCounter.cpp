#include "cir/gpuprocessing/GpuMomentCounter.h"
#include "cir/gpuprocessing/count_moments.cuh"
#include "cir/common/config.h"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuMomentCounter::GpuMomentCounter() {

}

GpuMomentCounter::~GpuMomentCounter() {
	count_raw_moment_shutdown();
}

void GpuMomentCounter::init(int width, int height) {
	count_raw_moment_init(width, height);
}

double* GpuMomentCounter::countHuMoments(MatWrapper& matWrapper) {
	cv::gpu::GpuMat mat = matWrapper.getGpuMat();
	uchar* data = mat.data;
	int width = mat.cols;
	int height = mat.rows;
	int step = mat.step;

	double M00 = count_raw_moment(data, width, height, step, 0, 0);
	double M01 = count_raw_moment(data, width, height, step, 0, 1);
	double M10 = count_raw_moment(data, width, height, step, 1, 0);
	double M11 = count_raw_moment(data, width, height, step, 1, 1);
	double M02 = count_raw_moment(data, width, height, step, 0, 2);
	double M20 = count_raw_moment(data, width, height, step, 2, 0);
	double M21 = count_raw_moment(data, width, height, step, 2, 1);
	double M12 = count_raw_moment(data, width, height, step, 1, 2);
	double M30 = count_raw_moment(data, width, height, step, 3, 0);
	double M03 = count_raw_moment(data, width, height, step, 0, 3);

	cv::Moments cvMoments(M00, M10, M01, M20, M11, M02, M30, M21, M12, M03);

	double* huMoments = (double*) malloc(sizeof(double) * HU_MOMENTS_NUMBER);

	cv::HuMoments(cvMoments, huMoments);

	return huMoments;
}

}}
