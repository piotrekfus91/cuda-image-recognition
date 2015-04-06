#include "cir/common/concurrency/VideoWriterThread.h"
#include <iostream>

using namespace cv;
using namespace cir::common;
using namespace cir::common::concurrency;

namespace cir { namespace common { namespace concurrency {

VideoWriterThread::VideoWriterThread(VideoWriter* videoWriter,
		IndexedMatWrapperBlockingQueue** postConversionQueues, int threadsNumber,
		ImageProcessingService* service, int frameRate, VideoReaderThread* reader) {
	_videoWriter = videoWriter;
	_postConversionQueues = postConversionQueues;
	_threadsNumber = threadsNumber;
	_service = service;
	_frameIdx = 0;
	_frameRate = frameRate;
	_reader = reader;
}

VideoWriterThread::~VideoWriterThread() {

}

void VideoWriterThread::operator()() {
	if(_videoWriter == NULL)
		cv::namedWindow("Video");

	while(true) {
		IndexedMatWrapper imw = readMatWrapper();
		if(imw.isPoison()) {
			break;
		}

		MatWrapper mw = imw.matWrapper;

		if(_videoWriter == NULL) {
			cv::imshow("Video", _service->getMat(mw));
			if(cv::waitKey(1000 / _frameRate) == 27) { // ESC
				_reader->stop();
			}
		} else {
			_videoWriter->write(_service->getMat(mw));
		}
	}
}

IndexedMatWrapper VideoWriterThread::readMatWrapper() {
	IndexedMatWrapperBlockingQueue* queue = _postConversionQueues[_frameIdx % _threadsNumber];
	IndexedMatWrapper imw = queue->get();
	_frameIdx++;
	return imw;
}

}}}
