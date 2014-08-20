#ifndef MATWRAPPER_H_
#define MATWRAPPER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpumat.hpp>

namespace cir { namespace common {

class MatWrapper {
public:
	MatWrapper(const cv::Mat& mat);
	MatWrapper(const cv::gpu::GpuMat& gpuMat);

	MatWrapper clone() const;

	cv::Mat getMat() const;
	cv::gpu::GpuMat getGpuMat() const;

	enum MAT_TYPE {
		MAT,
		GPU_MAT
	};

	MAT_TYPE getType() const;

private:
	cv::Mat _mat;
	cv::gpu::GpuMat _gpuMat;
	MAT_TYPE _matType;
};

}}

#endif
