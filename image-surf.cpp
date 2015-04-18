#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/cpuprocessing/CpuUnionFindSegmentator.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuUnionFindSegmentator.h"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/logger/BufferedConfigurableLogger.h"
#include "cir/common/logger/NullLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/recognition/MetroRecognizor.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::cpuprocessing;
using namespace cir::gpuprocessing;

int main(int argc, char** argv) {
	std::list<std::string> loggerConf;
	loggerConf.push_back("Perform SURF");
	loggerConf.push_back("SURF find matches");
	loggerConf.push_back("SURF similarity");
	BufferedConfigurableLogger logger(loggerConf);
	GpuImageProcessingService service(logger);
	MetroRecognizor recognizor(service);
	recognizor.learn(getTestFile("metro", "metro.png").c_str());

	cv::Mat mat1 = cv::imread(getTestFile("metro", "metro.png").c_str());
//	cv::Mat mat2 = cv::imread(getTestFile("metro", "metro.png").c_str());
//	cv::Mat mat2 = cv::imread(getTestFile("metro", "metro_scaled_rotated.png").c_str());
//	cv::Mat mat2 = cv::imread(getTestFile("metro", "metro_warszawa_450.jpeg").c_str());
	cv::Mat mat2 = cv::imread(getTestFile("metro", "metro-imielin.jpg").c_str());

	MatWrapper mw1 = service.getMatWrapper(mat1);
	MatWrapper mw2 = service.getMatWrapper(mat2);

	RecognitionInfo recognitionInfo1 = recognizor.recognize(mw1);
	RecognitionInfo recognitionInfo2 = recognizor.recognize(mw2);

	MatWrapper greyMw1 = service.toGrey(mw1);
	MatWrapper greyMw2 = service.toGrey(mw2);

	if(recognitionInfo1.isSuccess() && recognitionInfo2.isSuccess()) {
		SurfApi* surfApi = service.getSurfApi();
		SurfPoints surfPoints1 = surfApi->performSurf(greyMw1, 400);
		SurfPoints surfPoints2 = surfApi->performSurf(greyMw2, 400);

		std::vector<cv::DMatch> matches = surfApi->findMatches(surfPoints1, surfPoints2);

		for(int segm1idx = 0; segm1idx < recognitionInfo1.getMatchedSegments()->size; segm1idx++) {
			float bestMatch = -1;
			Segment* segm1 = recognitionInfo1.getMatchedSegments()->segments[segm1idx];

			for(int segm2idx = 0; segm2idx < recognitionInfo2.getMatchedSegments()->size; segm2idx++) {
				Segment* segm2 = recognitionInfo2.getMatchedSegments()->segments[segm2idx];

				float currentMatch = surfApi->getSimilarity(surfPoints1, segm1, surfPoints2, segm2, matches);
				if(currentMatch > bestMatch) {
					bestMatch = currentMatch;
				}
			}
			std::cout << bestMatch << std::endl;
		}
	}

	logger.flushBuffer();
}
