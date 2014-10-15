#include "opencv2/opencv.hpp"
#include "cir/cpuprocessing/CpuMomentCounter.h"
#include "cir/common/config.h"
#include <iostream>

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuMomentCounter::CpuMomentCounter() {

}

CpuMomentCounter::~CpuMomentCounter() {

}

double* CpuMomentCounter::countHuMoments(MatWrapper& matWrapper) {
	cv::Mat mat = matWrapper.getMat();
	uchar* data = mat.data;
	int width = mat.cols;
	int height = mat.rows;
	int step = mat.step;

	double M00 = countRawMoment(data, width, height, step, 0, 0);
	double M01 = countRawMoment(data, width, height, step, 0, 1);
	double M10 = countRawMoment(data, width, height, step, 1, 0);
	double M11 = countRawMoment(data, width, height, step, 1, 1);
	double M02 = countRawMoment(data, width, height, step, 0, 2);
	double M20 = countRawMoment(data, width, height, step, 2, 0);
	double M21 = countRawMoment(data, width, height, step, 2, 1);
	double M12 = countRawMoment(data, width, height, step, 1, 2);
	double M30 = countRawMoment(data, width, height, step, 3, 0);
	double M03 = countRawMoment(data, width, height, step, 0, 3);

	cv::Moments cvMoments(M00, M10, M01, M20, M11, M02, M30, M21, M12, M03);

	double* huMoments = (double*) malloc(sizeof(double) * HU_MOMENTS_NUMBER);
	cv::HuMoments(cvMoments, huMoments);
	return huMoments;
}

double CpuMomentCounter::countRawMoment(uchar* data, int width, int height, int step, int p, int q) {
	double sum = 0;
	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			int idx = y * step + x;
			double pixel = data[idx];

			double value = 0;
			if(p == 0 && q == 0) {
				value = pixel;
			} else if(q == 0) {
				value = pixel * pow(x, p);
			} else if(p == 0) {
				value = pixel * pow(y, q);
			} else {
				value = pixel * pow(x, p) * pow(y, q);
			}

			sum += value;
		}
	}

	return sum;
}

int CpuMomentCounter::pow(int p, int q) {
	if(q == 0)
		return 1;

	if(q == 1)
		return p;

	if(q == 2)
		return p * p;

	if(q == 3)
		return p * p * p;

	return -1; // should never happen!
}

}}
