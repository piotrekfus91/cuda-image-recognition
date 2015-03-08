#include "cir/common/concurrency/VideoWriterThread.h"
#include <iostream>

using namespace cv;
using namespace cir::common;
using namespace cir::common::concurrency;

namespace cir { namespace common { namespace concurrency {

VideoWriterThread::VideoWriterThread(VideoWriter* videoWriter,
		IndexedMatWrapperBlockingQueue** postConversionQueues, int threadsNumber,
		ImageProcessingService* service) {
	_videoWriter = videoWriter;
	_postConversionQueues = postConversionQueues;
	_threadsNumber = threadsNumber;
	_service = service;
	_frameIdx = 0;
}

VideoWriterThread::~VideoWriterThread() {

}

void VideoWriterThread::operator()() {
	while(true) {
		IndexedMatWrapper imw = readMatWrapper();
		if(imw.isPoison()) {
			break;
		}

		MatWrapper mw = imw.matWrapper;

//		cv::imshow("s", mw.getMat());
//		cv::waitKey(0);

		std::cerr << imw.id << std::endl;
		_videoWriter->write(mw.getMat());
	}
}

IndexedMatWrapper VideoWriterThread::readMatWrapper() {
	IndexedMatWrapperBlockingQueue* queue = _postConversionQueues[_frameIdx % _threadsNumber];
	IndexedMatWrapper imw = queue->get();
	_frameIdx++;
	return imw;
}

}}}
