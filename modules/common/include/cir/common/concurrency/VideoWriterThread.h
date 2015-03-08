#ifndef VIDEOWRITERTHREAD_H_
#define VIDEOWRITERTHREAD_H_

#include <opencv2/opencv.hpp>
#include "cir/common/ImageProcessingService.h"
#include "cir/common/concurrency/IndexedMatWrapperBlockingQueue.h"

namespace cir { namespace common { namespace concurrency {

class VideoWriterThread {
public:
	VideoWriterThread(cv::VideoWriter* videoWriter,
			cir::common::concurrency::IndexedMatWrapperBlockingQueue** postConversionQueues,
			int threadsNumber, cir::common::ImageProcessingService* service);
	virtual ~VideoWriterThread();

	void operator()();

private:
	int _threadsNumber;
	cir::common::concurrency::IndexedMatWrapperBlockingQueue** _postConversionQueues;
	cv::VideoWriter* _videoWriter;
	cir::common::ImageProcessingService* _service;
	int _frameIdx;
	cv::Mat _poison;

	cir::common::concurrency::IndexedMatWrapper readMatWrapper();
	bool isPoison(cir::common::MatWrapper& mw);
};

}}}
#endif /* VIDEOWRITERTHREAD_H_ */
