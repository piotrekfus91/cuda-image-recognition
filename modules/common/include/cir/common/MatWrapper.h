#ifndef MATWRAPPER_H_
#define MATWRAPPER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpumat.hpp>

namespace cir { namespace common {

class MatWrapper {
public:
	MatWrapper();
	MatWrapper(const cv::Mat& mat);
	MatWrapper(const cv::gpu::GpuMat& gpuMat);

	virtual ~MatWrapper();

	MatWrapper clone() const;

	cv::Mat getMat() const;
	cv::gpu::GpuMat getGpuMat() const;

	enum MAT_TYPE {
		NONE,
		MAT,
		GPU_MAT
	};

	enum COLOR_SCHEME {
		BGR,
		HSV,
		GRAY,
		UNKNOWN
	};

	MAT_TYPE getType() const;
	COLOR_SCHEME getColorScheme() const;
	void setColorScheme(const COLOR_SCHEME colorScheme);
	int getWidth() const;
	int getHeight() const;

private:
	cv::Mat _mat;
	cv::gpu::GpuMat _gpuMat;
	MAT_TYPE _matType;
	COLOR_SCHEME _colorScheme;

	void validateType();
};

}}

#endif
