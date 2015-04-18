#ifndef SURFAPI_H_
#define SURFAPI_H_

#include "cir/common/MatWrapper.h"
#include "cir/common/SurfPoints.h"
#include "cir/common/Segment.h"
#include "cir/common/logger/Logger.h"

namespace cir { namespace common {

class SurfApi {
public:
	SurfApi(cir::common::logger::Logger& logger);
	virtual ~SurfApi();

	virtual SurfPoints performSurf(MatWrapper& mw, int minHessian);
	virtual std::vector<cv::DMatch> findMatches(SurfPoints& surfPoints1,
			SurfPoints& surfPoints2);
	virtual float getSimilarity(SurfPoints& surfPoints1, Segment* segm1,
			SurfPoints& surfPoints2, Segment* segm2, std::vector<cv::DMatch> matches);

protected:
	cir::common::logger::Logger& _logger;

	virtual SurfPoints doPerformSurf(MatWrapper& mw, int minHessian) = 0;
	virtual std::vector<cv::DMatch> doFindMatches(SurfPoints& surfPoints1,
			SurfPoints& surfPoints2) = 0;
	virtual float doGetSimilarity(SurfPoints& surfPoints1, Segment* segm1,
			SurfPoints& surfPoints2, Segment* segm2, std::vector<cv::DMatch> matches);
};

}}
#endif /* SURFAPI_H_ */
