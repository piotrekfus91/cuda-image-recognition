#ifndef VIDEOREADERTHREAD_H_
#define VIDEOREADERTHREAD_H_

#include "cir/common/concurrency/IndexedMatWrapperBlockingQueue.h"
#include "cir/common/ImageProcessingService.h"
#include <opencv2/opencv.hpp>

namespace cir { namespace common { namespace concurrency {

class VideoReaderThread {
public:
	VideoReaderThread(cv::VideoCapture* videoReader,
			cir::common::concurrency::IndexedMatWrapperBlockingQueue** preConversionQueues,
			int threadsNumber, cir::common::ImageProcessingService* service);
	virtual ~VideoReaderThread();

	void operator()();

	void stop();

private:
	int _threadsNumber;
	cir::common::concurrency::IndexedMatWrapperBlockingQueue** _preConversionQueues;
	cv::VideoCapture* _videoReader;
	cir::common::ImageProcessingService* _service;
	int _frameIdx;
	bool run;

	void addMatWrapper(cir::common::concurrency::IndexedMatWrapper& imw);
	void addPoisonMatWrapper();
};

}}}
#endif /* VIDEOREADERTHREAD_H_ */
