#ifndef CPUSURFAPI_H_
#define CPUSURFAPI_H_

#include "cir/common/SurfApi.h"

namespace cir { namespace cpuprocessing {

class CpuSurfApi : public cir::common::SurfApi {
public:
	CpuSurfApi(cir::common::logger::Logger& logger);
	virtual ~CpuSurfApi();

protected:
	virtual cir::common::SurfPoints doPerformSurf(cir::common::MatWrapper& mw, int minHessian);
	virtual std::vector<cv::DMatch> doFindMatches(cir::common::SurfPoints& surfPoints1,
			cir::common::SurfPoints& surfPoints2);
	virtual float doGetSimilarity(cir::common::SurfPoints& surfPoints1, cir::common::Segment* segm1,
			cir::common::SurfPoints& surfPoints2, cir::common::Segment* segm2,
			std::vector<cv::DMatch> matches);
};

}
}
#endif /* CPUSURFAPI_H_ */
