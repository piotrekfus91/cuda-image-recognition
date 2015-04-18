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
	surfPoints.descriptors = descriptors;
	return surfPoints;
}

std::vector<cv::DMatch> CpuSurfApi::doFindMatches(SurfPoints& surfPoints1,
		SurfPoints& surfPoints2) {
	cv::BFMatcher matcher(cv::NORM_L2);
	std::vector<cv::DMatch> matches;
	matcher.match(surfPoints1.descriptors, surfPoints2.descriptors, matches);
	return matches;
}

float CpuSurfApi::doGetSimilarity(SurfPoints& surfPoints1, Segment* segm1,
		SurfPoints& surfPoints2, Segment* segm2, std::vector<cv::DMatch> matches) {
	std::vector<cv::KeyPoint> keyPoints1 = surfPoints1.keyPoints;
	std::vector<cv::KeyPoint> keyPoints2 = surfPoints2.keyPoints;

	float totalDistance = 0.;
	int hits = 0;

	for(std::vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end(); it++) {
		cv::DMatch match = *it;
		cv::KeyPoint keyPoint1 = keyPoints1[match.queryIdx];
		cv::KeyPoint keyPoint2 = keyPoints2[match.trainIdx];

		cv::Point2f point1 = keyPoint1.pt;
		cv::Point2f point2 = keyPoint2.pt;

		if(segm1->contains(point1.x, point1.y) && segm2->contains(point2.x, point2.y)) {
			totalDistance += match.distance;
			hits++;
		}
	}

	if(hits == 0)
		return -1;

	return totalDistance / hits;
}

}}
