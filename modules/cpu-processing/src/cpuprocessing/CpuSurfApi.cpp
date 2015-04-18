#include "cir/cpuprocessing/CpuSurfApi.h"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/gpu.hpp>

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuSurfApi::CpuSurfApi(cir::common::logger::Logger& logger) : SurfApi(logger) {

}

CpuSurfApi::~CpuSurfApi() {

}

SurfPoints CpuSurfApi::doPerformSurf(MatWrapper& mw, int minHessian) {
	cv::SURF surf(minHessian);
	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptors;

	surf(mw.getMat(), cv::Mat(), keyPoints, descriptors);

	SurfPoints surfPoints;
	surfPoints.keyPoints = keyPoints;
	surfPoints.descriptorsAsMat = descriptors;
	return surfPoints;
}

std::vector<cv::DMatch> CpuSurfApi::doFindMatches(SurfPoints& surfPoints1,
		SurfPoints& surfPoints2) {
	cv::BFMatcher matcher(cv::NORM_L2);
	std::vector<cv::DMatch> matches;
	matcher.match(surfPoints1.descriptorsAsMat, surfPoints2.descriptorsAsMat, matches);
	return matches;
}

}}
