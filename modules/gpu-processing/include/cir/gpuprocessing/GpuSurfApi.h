#ifndef GPUSURFAPI_H_
#define GPUSURFAPI_H_

#include "cir/common/SurfApi.h"

namespace cir { namespace gpuprocessing {

class GpuSurfApi : public cir::common::SurfApi {
public:
	GpuSurfApi(cir::common::logger::Logger& logger);
	virtual ~GpuSurfApi();

protected:
	virtual cir::common::SurfPoints doPerformSurf(cir::common::MatWrapper& mw, int minHessian);
	virtual std::vector<cv::DMatch> doFindMatches(cir::common::SurfPoints& surfPoints1,
			cir::common::SurfPoints& surfPoints2);
};

}}
#endif /* GPUSURFAPI_H_ */
