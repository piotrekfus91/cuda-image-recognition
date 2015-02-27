#include "cir/common/video/VideoHandler.h"

namespace cir { namespace common { namespace video {

VideoHandler::VideoHandler() {

}

VideoHandler::~VideoHandler() {

}

void VideoHandler::handle(std::string& inputFilePath, VideoConverter* converter) {
	int extensionStart = inputFilePath.find_last_of(".");
	std::string extension = inputFilePath.substr(extensionStart);
	std::string outputPath = inputFilePath.substr(0, extensionStart);
	outputPath.append("_out");
	outputPath.append(extension);
	handle(inputFilePath, outputPath, converter);
}

}}}
