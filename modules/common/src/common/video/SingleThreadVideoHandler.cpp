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

void SingleThreadVideoHandler::handle(cv::VideoCapture* videoReader, cv::VideoWriter* videoWriter,
		VideoConverter* converter, int frameRate) {
	cv::namedWindow("Video");
	videoReader->set(CV_CAP_PROP_FPS, frameRate);
	videoReader->set(CV_CAP_PROP_FRAME_WIDTH, 640);
	videoReader->set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	bool initialized = false;
	Mat frame;
	int frameIdx = 0;

	int frameTime = frameRate != 0 ? 1000 / frameRate : 0;

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
			clock_t endTime = clock();
			double totalTime = double(endTime - startTime) / CLOCKS_PER_SEC * 1000;
			int timeToWait = frameTime - totalTime < 1 ? 1 : frameTime - totalTime;
			std::cerr << totalTime << " " << timeToWait << std::endl;
			if(cv::waitKey(timeToWait) == 27) { // ESC
				break;
			}
		}

		frameIdx++;
	}

	videoReader->release();
	if(videoWriter != NULL)
		videoWriter->release();
}

void SingleThreadVideoHandler::handle(std::string& inputFilePath, std::string& outputFilePath,
		VideoConverter* converter) {
	VideoCapture videoReader = openVideoReader(inputFilePath);
	VideoWriter videoWriter = openVideoWriter(videoReader, outputFilePath);
	handle(&videoReader, &videoWriter, converter, -1);
}

void SingleThreadVideoHandler::handle(int cameraIdx, VideoConverter* converter, int frameRate) {
	VideoCapture videoReader = openVideoReader(cameraIdx);
	handle(&videoReader, NULL, converter, frameRate);
}

}}}
