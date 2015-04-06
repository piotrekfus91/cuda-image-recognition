#include "cir/common/video/MultiThreadVideoHandler.h"
#include "cir/common/concurrency/VideoReaderThread.h"
#include "cir/common/concurrency/VideoWriterThread.h"
#include "cir/common/concurrency/ConversionThread.h"
#include "cir/devenv/ThreadInfo.h"
#include "cir/common/config.h"
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>

#include "cir/common/recognition/RegistrationPlateRecognizor.h"

using namespace cv;
using namespace cir::common;
using namespace cir::common::concurrency;

namespace cir { namespace common { namespace video {

MultiThreadVideoHandler::MultiThreadVideoHandler() {
	_threadNumber = cir::devenv::ThreadInfo::getNumberOfThreads();
	_preConversionQueues = new IndexedMatWrapperBlockingQueue*[_threadNumber];
	_postConversionQueues = new IndexedMatWrapperBlockingQueue*[_threadNumber];

	for(int i = 0; i < _threadNumber; i++) {
		_preConversionQueues[i] = new IndexedMatWrapperBlockingQueue(MAT_WRAPPER_BLOCKING_QUEUE_SIZE);
		_postConversionQueues[i] = new IndexedMatWrapperBlockingQueue(MAT_WRAPPER_BLOCKING_QUEUE_SIZE);
	}
}

MultiThreadVideoHandler::~MultiThreadVideoHandler() {

}

void MultiThreadVideoHandler::handle(std::string& inputFilePath, std::string& outputFilePath,
		VideoConverter* converter) {
	VideoCapture videoReader = openVideoReader(inputFilePath);
	VideoWriter videoWriter = openVideoWriter(videoReader, outputFilePath);

	handle(&videoReader, &videoWriter, converter, 0);
}

void MultiThreadVideoHandler::handle(int cameraIdx, VideoConverter* converter, int frameRate) {
	cv::VideoCapture videoReader = openVideoReader(cameraIdx);
	videoReader.set(CV_CAP_PROP_FPS, frameRate);
	videoReader.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	videoReader.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	handle(&videoReader, NULL, converter, frameRate);
}

void MultiThreadVideoHandler::handle(cv::VideoCapture* videoReader, cv::VideoWriter* videoWriter,
		VideoConverter* converter, int frameRate) {
	VideoReaderThread reader(videoReader, _preConversionQueues, _threadNumber,
			converter->getService());
	VideoWriterThread writer(videoWriter, _postConversionQueues, _threadNumber,
			converter->getService(), frameRate, &reader);

	boost::thread readerThread(reader);
	boost::thread writerThread(writer);

	for(int i = 0; i < _threadNumber; i++) {
		ConversionThread conversion(converter, _preConversionQueues[i],
				_postConversionQueues[i]);
		boost::thread conversionThread(conversion);
	}

	writerThread.join();

	videoReader->release();
	if(videoWriter != NULL)
		videoWriter->release();
}

}}}
