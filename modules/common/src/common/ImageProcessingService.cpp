#include "cir/common/ImageProcessingService.h"

using namespace cir::common;

int ImageProcessingService::DEFAULT_LOW_PASS_KERNEL_SIZE = 3;
cv::Mat ImageProcessingService::DEFAULT_LOW_PASS_KERNEL =
		cv::Mat::ones(DEFAULT_LOW_PASS_KERNEL_SIZE, DEFAULT_LOW_PASS_KERNEL_SIZE, CV_32F)
				/ (float)(DEFAULT_LOW_PASS_KERNEL_SIZE*DEFAULT_LOW_PASS_KERNEL_SIZE);

ImageProcessingService::ImageProcessingService() {

}

ImageProcessingService::~ImageProcessingService() {
}
