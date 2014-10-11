#include "cir/common/ImageProcessingService.h"
#include "opencv2/opencv.hpp"

using namespace cir::common;

int ImageProcessingService::DEFAULT_LOW_PASS_KERNEL_SIZE = 3;
cv::Mat ImageProcessingService::DEFAULT_LOW_PASS_KERNEL =
		cv::Mat::ones(DEFAULT_LOW_PASS_KERNEL_SIZE, DEFAULT_LOW_PASS_KERNEL_SIZE, CV_32F)
				/ (float)(DEFAULT_LOW_PASS_KERNEL_SIZE*DEFAULT_LOW_PASS_KERNEL_SIZE);

ImageProcessingService::ImageProcessingService() {

}

ImageProcessingService::~ImageProcessingService() {

}

void ImageProcessingService::loadPattern(std::string filePath) {
	cv::Mat mat = cv::imread(filePath, CV_LOAD_IMAGE_UNCHANGED);
	MatWrapper mw(mat);

	// TODO fileName
	std::string fileName;
	int fileNameStart = filePath.find_last_of('/');
	if(fileNameStart == std::string::npos)
		fileName = filePath;
	else
		fileName = fileName.substr(fileNameStart + 1);

	double* huMoments = countHuMoments(mw);
	Pattern* pattern = new Pattern(fileName, 1, &huMoments);
	patterns[filePath] = pattern;
}
