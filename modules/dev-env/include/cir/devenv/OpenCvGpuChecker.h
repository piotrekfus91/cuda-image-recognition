#ifndef OPENCVGPUCHECKER_H_
#define OPENCVGPUCHECKER_H_

namespace cir { namespace devenv {

class OpenCvGpuChecker {
public:
	OpenCvGpuChecker();
	virtual ~OpenCvGpuChecker();

	bool canRunOpenCvGpuCode();
	int getOpenCvGpuDeviceCount();
};

}}
#endif
