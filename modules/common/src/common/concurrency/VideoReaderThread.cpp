#include "cir/common/concurrency/VideoReaderThread.h"
#include <iostream>

using namespace cv;
using namespace cir::common;
using namespace cir::common::concurrency;

namespace cir { namespace common { namespace concurrency {

VideoReaderThread::VideoReaderThread(VideoCapture* videoReader,
		IndexedMatWrapperBlockingQueue** preConversionQueues, int threadsNumber,
		ImageProcessingService* service) {
	_threadsNumber = threadsNumber;
	_preConversionQueues = preConversionQueues;
	_videoReader = videoReader;
	_service = service;
	_frameIdx = 0;
}

VideoReaderThread::~VideoReaderThread() {

}

void VideoReaderThread::operator()() {
	Mat frame;
//	cv::namedWindow("r");
	while(true) {
		bool frameRead = _videoReader->read(frame);
		if(!frameRead) {
			addPoisonMatWrapper();
			break;
		}

//		cv::imshow("r", frame);
//		cv::waitKey(0);

		MatWrapper mw = _service->getMatWrapper(frame);
		IndexedMatWrapper imw;
		imw.matWrapper = mw;
		imw.id = _frameIdx;
		addMatWrapper(imw);
	}
}

void VideoReaderThread::addMatWrapper(IndexedMatWrapper& imw) {
	IndexedMatWrapperBlockingQueue* queue = _preConversionQueues[_frameIdx % _threadsNumber];
	queue->add(imw);
	_frameIdx++;
}

void VideoReaderThread::addPoisonMatWrapper() {
	IndexedMatWrapper imw;
	imw.id = -1;
	for(int i = 0; i < _threadsNumber; i++)
		addMatWrapper(imw);
}

}}}
