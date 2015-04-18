#include "cir/gpuprocessing/GpuSurfApi.h"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cir::common;

namespace cir { namespace gpuprocessing {

GpuSurfApi::GpuSurfApi(cir::common::logger::Logger& logger) : SurfApi(logger) {

}

GpuSurfApi::~GpuSurfApi() {

}

SurfPoints GpuSurfApi::doPerformSurf(MatWrapper& mw, int minHessian) {
	cv::gpu::SURF_GPU surf(minHessian);
	cv::gpu::GpuMat d_keyPoints;
	cv::gpu::GpuMat d_descriptors;

	surf(mw.getGpuMat(), cv::gpu::GpuMat(), d_keyPoints, d_descriptors);

	std::vector<cv::KeyPoint> keyPoints;
	surf.downloadKeypoints(d_keyPoints, keyPoints);

	SurfPoints surfPoints;
	surfPoints.keyPoints = keyPoints;
	surfPoints.descriptorsAsGpuMat = d_descriptors;
	return surfPoints;
}

std::vector<cv::DMatch> GpuSurfApi::doFindMatches(SurfPoints& surfPoints1,
		SurfPoints& surfPoints2) {
	cv::gpu::BFMatcher_GPU matcher(cv::NORM_L2);

	cv::gpu::GpuMat trainIdx, distance;
	std::vector<cv::DMatch> matches;

	matcher.matchSingle(surfPoints1.descriptorsAsGpuMat, surfPoints2.descriptorsAsGpuMat,
			trainIdx, distance);
	cv::gpu::BFMatcher_GPU::matchDownload(trainIdx, distance, matches);
	return matches;
}

}}
