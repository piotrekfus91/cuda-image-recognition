#include "ConfigHelper.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "cir/common/ImageProcessingService.h"
#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/common/recognition/Recognizor.h"
#include "cir/common/recognition/MetroRecognizor.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "cir/common/video/VideoHandler.h"
#include "cir/common/video/SingleThreadVideoHandler.h"
#include "cir/common/video/MultiThreadVideoHandler.h"
#include "cir/common/logger/Logger.h"
#include "cir/common/test_file_loader.h"

using namespace cir::common;
using namespace cir::common::recognition;
using namespace cir::common::video;
using namespace cir::common::logger;
using namespace cir::cpuprocessing;
using namespace cir::gpuprocessing;

ConfigHelper::ConfigHelper(int argc, char** argv) {
	for(int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		unsigned int separatorIndex = arg.find('=');
		if(arg.find('=') < 0 || separatorIndex >= arg.size()) {
			std::cerr << "not a valid config entry: " << arg << std::endl;
			exit(EXIT_FAILURE);
		}

		std::string key = arg.substr(0, separatorIndex);
		std::string value = arg.substr(separatorIndex + 1);

		_config[key] = value;
	}
}

ConfigHelper::~ConfigHelper() {

}

ImageProcessingService* ConfigHelper::getService(Logger& logger) {
	if(_config.find("service") == _config.end()) {
		std::cerr << "no service chosen, available are CPU and GPU only";
		exit(EXIT_FAILURE);
	}

	if(_config["service"] == "CPU")
		return new CpuImageProcessingService(logger);
	else if(_config["service"] == "GPU")
		return new GpuImageProcessingService(logger);
	else {
		std::cerr << "no service chosen, available are CPU and GPU only";
		exit(EXIT_FAILURE);
	}
}

Recognizor* ConfigHelper::getRecognizor(ImageProcessingService* service) {
	if(_config.find("recognizor") == _config.end()) {
		std::cerr << "no recognizor chosen, available are metro and plate only";
		exit(EXIT_FAILURE);
	}

	if(_config["recognizor"] == "metro") {
		MetroRecognizor* recognizor	= new MetroRecognizor(*service);
		recognizor->learn(getTestFile("metro", "metro.png").c_str());
		service->setSegmentatorMinSize(30);
		return recognizor;
	} else if(_config["recognizor"] == "plate") {
		RegistrationPlateRecognizor* recognizor = new RegistrationPlateRecognizor(*service);
		recognizor->setWriteLetters(isWriteLetters());
		return recognizor;
	} else {
		std::cerr << "no recognizor chosen, available are metro and plate only";
		exit(EXIT_FAILURE);
	}
}
VideoHandler* ConfigHelper::getVideoHandler() {
	if(_config.find("thread") == _config.end()) {
		std::cerr << "no thread specified, available are single and multi only";
		exit(EXIT_FAILURE);
	}

	if(_config["thread"] == "single")
		return new SingleThreadVideoHandler();
	else if(_config["thread"] == "multi")
		return new MultiThreadVideoHandler();
	else {
		std::cerr << "no thread specified, available are single and multi only";
		exit(EXIT_FAILURE);
	}
}
int ConfigHelper::getFrameRate() {
	if(_config.find("frameRate") == _config.end()) {
		std::cerr << "no frameRate specified";
		exit(EXIT_FAILURE);
	}

	return atoi(_config["frameRate"].c_str());
}
std::string ConfigHelper::getVideoFilePath() {
	return getFilePath("video");
}

std::string ConfigHelper::getMetroFilePath() {
	return getFilePath("metro");
}

std::string ConfigHelper::getPlateFilePath() {
	return getFilePath("registration-plate");
}

std::string ConfigHelper::getFilePath(const char* library) {
	if(_config.find("file") == _config.end()) {
		std::cerr << "no file specified";
		exit(EXIT_FAILURE);
	}

	return cir::common::getTestFile(library, _config["file"]);
}

std::string ConfigHelper::getMode(std::string& availableModes) {
	if(_config.find("mode") == _config.end()) {
		std::cerr << "no mode specified, available are " << availableModes << " only";
		exit(EXIT_FAILURE);
	}

	return _config["mode"];
}

bool ConfigHelper::isWriteLetters() {
	if(_config.find("writeLetters") == _config.end())
		return false;

	return _config["writeLetters"] == "true";
}

bool ConfigHelper::withSurf() {
	if(_config.find("withSurf") == _config.end())
		return false;

	return _config["withSurf"] == "true";
}
