#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <boost/thread.hpp>
#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/cpuprocessing/CpuUnionFindSegmentator.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuUnionFindSegmentator.h"
#include "cir/common/cuda_host_util.cuh"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/logger/NullLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/video/SingleThreadVideoHandler.h"
#include "cir/common/video/MultiThreadVideoHandler.h"
#include "cir/common/video/RecognitionVideoConverter.h"
#include "cir/common/recognition/MetroRecognizor.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "cir/common/recognition/RegistrationPlateTeacher.h"
#include "cir/devenv/ThreadInfo.h"
#include "ConfigHelper.h"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::common::video;
using namespace cir::cpuprocessing;
using namespace cir::gpuprocessing;

int main(int argc, char** argv) {
	ConfigHelper config = ConfigHelper(argc, argv);

	NullLogger logger;

	ImageProcessingService* service = config.getService(logger);
	Recognizor* recognizor = config.getRecognizor(service);
	VideoHandler* videoHandler = config.getVideoHandler();

	RecognitionVideoConverter* videoConverter = new RecognitionVideoConverter(recognizor, service);
	if(config.withSurf())
		videoConverter->withSurf();

	std::string availableModes = "video and camera";

	if(config.getMode(availableModes) == "camera") {
		videoHandler->handle(0, videoConverter, config.getFrameRate());
		return EXIT_SUCCESS;
	}

	if(config.getMode(availableModes) == "video")  {
		std::string filePath = config.getVideoFilePath();
		videoHandler->handle(filePath, videoConverter);
		return EXIT_SUCCESS;
	}

	std::cerr << "available modes: " << availableModes;

	return EXIT_FAILURE;
}
