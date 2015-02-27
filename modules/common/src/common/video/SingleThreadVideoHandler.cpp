#include <iostream>
#include "cir/common/video/SingleThreadVideoHandler.h"
#include "cir/common/exception/VideoException.h"
#include <opencv2/opencv.hpp>
#include <ctime>

using namespace cv;
using namespace cir::common;
using namespace cir::common::exception;

namespace cir { namespace common { namespace video {

SingleThreadVideoHandler::SingleThreadVideoHandler() {

}

SingleThreadVideoHandler::~SingleThreadVideoHandler() {

}

void SingleThreadVideoHandler::handle(std::string& inputFilePath, std::string& outputFilePath,
		VideoConverter* converter) {
	VideoCapture videoReader(inputFilePath);
	if(!videoReader.isOpened())
		throw new VideoException("cannot open input video");

	double fourcc = videoReader.get(CV_CAP_PROP_FOURCC);
	double fps = videoReader.get(CV_CAP_PROP_FPS);
	double frameWidth = videoReader.get(CV_CAP_PROP_FRAME_WIDTH);
	double frameHeight = videoReader.get(CV_CAP_PROP_FRAME_HEIGHT);
	Size frameSize(frameWidth, frameHeight);

	VideoWriter videoWriter(outputFilePath, fourcc, fps, frameSize);
	if(!videoWriter.isOpened())
		throw new VideoException("cannot open output video");

	Mat frame;
	while (true) {
		bool frameRead = videoReader.read(frame);
		if(!frameRead)
			break;

		if(frame.type() != 0) {
			clock_t begin = clock();
			MatWrapper mw(frame);
			mw = converter->convert(mw);
			clock_t end = clock();
			std::cout << "frame time: " << double(end - begin) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
			videoWriter.write(mw.getMat());
		} else {
			videoWriter.write(frame);
		}

	}

	videoReader.release();
	videoWriter.release();
}

}}}
