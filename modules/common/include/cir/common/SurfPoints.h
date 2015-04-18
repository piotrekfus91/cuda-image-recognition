#ifndef SURFPOINTS_H_
#define SURFPOINTS_H_

#include <vector>
#include <opencv2/opencv.hpp>

namespace cir { namespace common {

class SurfPoints {
public:
	SurfPoints();
	virtual ~SurfPoints();

	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptors;
};

}}
#endif /* SURFPOINTS_H_ */
