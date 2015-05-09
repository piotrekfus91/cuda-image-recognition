#include <iostream>
#include "cir/common/video/SingleThreadVideoHandler.h"
#include "cir/common/exception/VideoException.h"
#include <opencv2/opencv.hpp>
#include <ctime>
#include <boost/chrono.hpp>

using namespace cv;
using namespace cir::common;
using namespace cir::common::exception;

namespace cir { namespace common { namespace video {

SingleThreadVideoHandler::SingleThreadVideoHandler() {

}

SingleThreadVideoHandler::~SingleThreadVideoHandler() {

}

void SingleThreadVideoHandler::handle(cv::VideoCapture* videoReader, cv::VideoWriter* videoWriter,
		VideoConverter* converter, int frameRate) {
	cv::namedWindow("Video");

	bool initialized = false;
	Mat frame;
	int frameIdx = 0;

	boost::chrono::high_resolution_clock::time_point start = boost::chrono::high_resolution_clock::now();

	while(true) {
		clock_t startTime = clock();
		bool frameRead = videoReader->read(frame);
		if(!frameRead)
			break;

		cv::Mat outMat;

		if(frame.type() != 0) {
			if(!initialized) {
				converter->getService()->init(frame.cols, frame.rows);
				initialized = true;
			}
			MatWrapper mw = converter->getService()->getMatWrapper(frame);
			mw = converter->convert(mw);

			if(mw.getType() == MatWrapper::MAT) {
				outMat = mw.getMat();
			} else {
				cv::gpu::GpuMat outGpuMat = mw.getGpuMat();
				outGpuMat.download(outMat);
			}
		} else {
			outMat = frame;
		}

		if(videoWriter != NULL) {
			videoWriter->write(outMat);
		} else {
			cv::imshow("Video", outMat);
			int timeToWait = countTimeToWait(startTime, frameRate);
			if(cv::waitKey(timeToWait) > 0) {
				break;
			}
		}

		frameIdx++;
	}

	videoReader->release();
	if(videoWriter != NULL)
		videoWriter->release();

	boost::chrono::high_resolution_clock::time_point end = boost::chrono::high_resolution_clock::now();
	boost::chrono::nanoseconds totalTimeInNano = end - start;

	int totalTimeInSec = totalTimeInNano.count() / 1000000000;
	int totalTimeInMillis = totalTimeInNano.count() / 1000000;

	std::cerr << "total time: " << totalTimeInSec << std::endl;
	std::cerr << "frames: " << frameIdx << std::endl;
	std::cerr << "avg time per frame: " << totalTimeInMillis / frameIdx << "s" << std::endl;
}

void SingleThreadVideoHandler::handle(std::string& inputFilePath, std::string& outputFilePath,
		VideoConverter* converter) {
	VideoCapture videoReader = openVideoReader(inputFilePath);
	VideoWriter videoWriter = openVideoWriter(videoReader, outputFilePath);
	handle(&videoReader, &videoWriter, converter, -1);
}

void SingleThreadVideoHandler::handle(int cameraIdx, VideoConverter* converter, int frameRate) {
	VideoCapture videoReader = openVideoReader(cameraIdx);
	videoReader.set(CV_CAP_PROP_FPS, frameRate);
	videoReader.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	videoReader.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	handle(&videoReader, NULL, converter, frameRate);
}

}}}
