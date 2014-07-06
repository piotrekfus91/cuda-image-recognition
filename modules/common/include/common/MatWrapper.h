#ifndef MATWRAPPER_H_
#define MATWRAPPER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpumat.hpp>

namespace cir { namespace common {

class MatWrapper {
public:
	MatWrapper(const cv::Mat& mat);
	MatWrapper(const cv::gpu::GpuMat& gpuMat);

	cv::Mat getMat();
	cv::gpu::GpuMat getGpuMat();

	enum MAT_TYPE {
		MAT,
		GPU_MAT
	};

	MAT_TYPE getType();

private:
	cv::Mat _mat;
	cv::gpu::GpuMat _gpuMat;
	MAT_TYPE _matType;
};

}}

#endif
