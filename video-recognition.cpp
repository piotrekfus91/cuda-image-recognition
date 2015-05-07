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
#include "cir/common/recognition/MetroRecognizor.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "cir/common/recognition/RegistrationPlateTeacher.h"
#include "cir/devenv/ThreadInfo.h"

using namespace std;
using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::common::video;
using namespace cir::cpuprocessing;
using namespace cir::gpuprocessing;

int main(int argc, char** argv) {
	NullLogger logger;
	GpuImageProcessingService cpuService(logger);
	cpuService.setSegmentator(new GpuUnionFindSegmentator);
//	CpuImageProcessingService cpuService(logger);
//	cpuService.setSegmentator(new CpuUnionFindSegmentator);
	cpuService.setSegmentatorMinSize(10);

//	RegistrationPlateRecognizor* recognizor = new RegistrationPlateRecognizor(cpuService);
//	RegistrationPlateTeacher teacher(recognizor);
//	teacher.teach(getTestFile("registration-plate", "alphabet"));
	MetroRecognizor* recognizor	= new MetroRecognizor(cpuService);
	recognizor->learn(getTestFile("metro", "metro.png").c_str());

//	VideoHandler* videoHandler = new SingleThreadVideoHandler();
	VideoHandler* videoHandler = new MultiThreadVideoHandler();
	RecognitionVideoConverter* videoConverter = new RecognitionVideoConverter(recognizor, &cpuService);
	std::string inputFilePath = getTestFile("video", "metro.avi");
	videoConverter->withSurf();
	videoHandler->handle(inputFilePath, videoConverter);

    return EXIT_SUCCESS;
}

