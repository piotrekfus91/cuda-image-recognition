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
	VideoCapture videoReader = openVideoReader(inputFilePath);
	VideoWriter videoWriter = openVideoWriter(videoReader, outputFilePath);
	int framesCount =  videoReader.get(CV_CAP_PROP_FRAME_COUNT);

	clock_t videoBegin = clock();

	Mat frame;
	while(true) {
		bool frameRead = videoReader.read(frame);
		if(!frameRead)
			break;

		if(frame.type() != 0) {
			clock_t begin = clock();
			MatWrapper mw = converter->getService()->getMatWrapper(frame);
			mw = converter->convert(mw);
			clock_t end = clock();
			std::cout << "frame time (" << int(videoReader.get(CV_CAP_PROP_POS_FRAMES)) << "): " <<
					double(end - begin) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
			videoWriter.write(mw.getMat());
		} else {
			videoWriter.write(frame);
		}

	}

	videoReader.release();
	videoWriter.release();

	clock_t videoEnd = clock();
	double videoTime = videoEnd - videoBegin;

	std::cout << "video time: " << videoTime / CLOCKS_PER_SEC << "s, frames count: " << framesCount << std::endl;
	std::cout << "avg frame time: " << int(videoTime / framesCount * 1000 / CLOCKS_PER_SEC) << "ms" << std::endl;
}

}}}
