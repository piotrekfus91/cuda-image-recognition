#ifndef SURFPOINTS_H_
#define SURFPOINTS_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpumat.hpp>

namespace cir { namespace common {

class SurfPoints {
public:
	SurfPoints();
	virtual ~SurfPoints();

	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptorsAsMat;
	cv::gpu::GpuMat descriptorsAsGpuMat;
};

}}
#endif /* SURFPOINTS_H_ */
