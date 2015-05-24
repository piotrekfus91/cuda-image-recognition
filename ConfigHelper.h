#ifndef CONFIGHELPER_H_
#define CONFIGHELPER_H_

#include "cir/common/ImageProcessingService.h"
#include "cir/common/recognition/Recognizor.h"
#include "cir/common/video/VideoHandler.h"
#include "cir/common/logger/Logger.h"

class ConfigHelper {
public:
	ConfigHelper(int argc, char** argv);
	virtual ~ConfigHelper();

	cir::common::ImageProcessingService* getService(cir::common::logger::Logger& logger);
	cir::common::recognition::Recognizor* getRecognizor(cir::common::ImageProcessingService* service);
	cir::common::video::VideoHandler* getVideoHandler();
	int getFrameRate();
	std::string getVideoFilePath();
	std::string getMetroFilePath();
	std::string getPlateFilePath();
	std::string getMode(std::string& availableModes);
	bool isWriteLetters();
	bool withSurf();
	bool keyExists(std::string key);

private:
	std::map<std::string, std::string> _config;
	std::string getFilePath(const char* library);
};

#endif /* CONFIGHELPER_H_ */
