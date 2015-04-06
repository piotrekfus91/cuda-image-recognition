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

VideoCapture VideoHandler::openVideoReader(int cameraIdx) const {
	VideoCapture videoReader(cameraIdx);
	if(!videoReader.isOpened())
		throw new VideoException("cannot open camera");
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

int VideoHandler::countTimeToWait(clock_t startTime, int frameRate) const {
	int frameTime = frameRate != 0 ? 1000 / frameRate : 0;
	clock_t endTime = clock();
	double totalTime = double(endTime - startTime) / CLOCKS_PER_SEC * 1000;
	int timeToWait = frameTime - totalTime < 1 ? 1 : frameTime - totalTime;
	return timeToWait;
}

}}}
