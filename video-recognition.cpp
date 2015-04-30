#include <cstdlib>
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
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "cir/common/recognition/RegistrationPlateTeacher.h"
#include "cir/devenv/ThreadInfo.h"

using namespace std;

int main(int argc, char** argv) {
	cir::common::logger::NullLogger logger;
	cir::gpuprocessing::GpuImageProcessingService cpuService(logger);
	cpuService.setSegmentator(new cir::gpuprocessing::GpuUnionFindSegmentator);
//	cir::cpuprocessing::CpuImageProcessingService cpuService(logger);
//	cpuService.setSegmentator(new cir::cpuprocessing::CpuUnionFindSegmentator);

	cir::common::recognition::RegistrationPlateRecognizor* recognizor
			= new cir::common::recognition::RegistrationPlateRecognizor(cpuService);
	cir::common::recognition::RegistrationPlateTeacher teacher(recognizor);
	teacher.teach(cir::common::getTestFile("registration-plate", "alphabet"));

	cir::common::video::VideoHandler* videoHandler = new cir::common::video::SingleThreadVideoHandler();
//	cir::common::video::VideoHandler* videoHandler = new cir::common::video::MultiThreadVideoHandler();
	cir::common::video::RecognitionVideoConverter* videoConverter
			= new cir::common::video::RecognitionVideoConverter(recognizor, &cpuService);
	std::string inputFilePath = cir::common::getTestFile("video", "walk.avi");
	videoConverter->withSurf();
	videoHandler->handle(inputFilePath, videoConverter);

    return EXIT_SUCCESS;
}

