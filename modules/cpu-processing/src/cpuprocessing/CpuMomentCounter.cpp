#include "opencv2/opencv.hpp"
#include "cir/cpuprocessing/CpuMomentCounter.h"
#include "cir/common/config.h"

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuMomentCounter::CpuMomentCounter() {

}

CpuMomentCounter::~CpuMomentCounter() {

}

double* CpuMomentCounter::countHuMoments(MatWrapper& matWrapper) {
	cv::Moments cvMoments = cv::moments(matWrapper.getMat());
	double* huMoments = (double*) malloc(sizeof(double) * HU_MOMENTS_NUMBER);
	cv::HuMoments(cvMoments, huMoments);
	return huMoments;
}

}}
