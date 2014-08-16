#ifndef CPUCOLORDETECTOR_H_
#define CPUCOLORDETECTOR_H_

#include "opencv2/opencv.hpp"

namespace cir { namespace cpuprocessing {

class CpuColorDetector {
public:
	CpuColorDetector();
	virtual ~CpuColorDetector();

	cv::Mat detectColor(cv::Mat& input, const int minHue, const int maxHue,
			const int minSat, const int maxSat, const int minValue, const int maxValue);
};

}}
#endif
