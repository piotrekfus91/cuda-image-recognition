#include "cir/common/SurfApi.h"
#include <ctime>

namespace cir { namespace common {

SurfApi::SurfApi(cir::common::logger::Logger& logger) : _logger(logger) {

}

SurfApi::~SurfApi() {

}

SurfPoints SurfApi::performSurf(MatWrapper& mw, int minHessian) {
	clock_t start = clock();
	SurfPoints surfPoints = doPerformSurf(mw, minHessian);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("Perform SURF", elapsed_secs);
	return surfPoints;
}

std::vector<cv::DMatch> SurfApi::findMatches(SurfPoints& surfPoints1,
		SurfPoints& surfPoints2) {
	clock_t start = clock();
	std::vector<cv::DMatch> matches = doFindMatches(surfPoints1, surfPoints2);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("SURF find matches", elapsed_secs);
	return matches;
}

float SurfApi::getSimilarity(SurfPoints& surfPoints1, Segment* segm1,
		SurfPoints& surfPoints2, Segment* segm2, std::vector<cv::DMatch> matches) {
	clock_t start = clock();
	float similarity = doGetSimilarity(surfPoints1, segm1, surfPoints2, segm2, matches);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("SURF similarity", elapsed_secs);
	return similarity;
}

float SurfApi::doGetSimilarity(SurfPoints& surfPoints1, Segment* segm1,
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
