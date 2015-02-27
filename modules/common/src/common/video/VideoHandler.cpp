#include "cir/common/video/VideoHandler.h"
#include "cir/common/exception/VideoException.h"

using namespace cv;
using namespace cir::common::exception;

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

VideoCapture VideoHandler::openVideoReader(std::string& inputFilePath) const {
	VideoCapture videoReader(inputFilePath);
	if(!videoReader.isOpened())
		throw new VideoException("cannot open input video");
	return videoReader;
}

VideoWriter VideoHandler::openVideoWriter(VideoCapture& videoReader, std::string& outputFilePath) const {
	double fourcc = videoReader.get(CV_CAP_PROP_FOURCC);
	double fps = videoReader.get(CV_CAP_PROP_FPS);
	double frameWidth = videoReader.get(CV_CAP_PROP_FRAME_WIDTH);
	double frameHeight = videoReader.get(CV_CAP_PROP_FRAME_HEIGHT);
	Size frameSize(frameWidth, frameHeight);

	VideoWriter videoWriter(outputFilePath, fourcc, fps, frameSize);
	if(!videoWriter.isOpened())
		throw new VideoException("cannot open output video");
	return videoWriter;
}

}}}
