#include <opencv2/opencv.hpp>
#include "cir/common/concurrency/ConversionThread.h"

using namespace cir::common;
using namespace cir::common::video;
using namespace cir::common::concurrency;

namespace cir { namespace common { namespace concurrency {

ConversionThread::ConversionThread(VideoConverter* converter,
		IndexedMatWrapperBlockingQueue* preConversionQueue,
		IndexedMatWrapperBlockingQueue* postConversionQueue) {
	_converter = converter;
	_preConversionQueue = preConversionQueue;
	_postConversionQueue = postConversionQueue;
}

ConversionThread::~ConversionThread() {

}

void ConversionThread::operator()() {
	while(true) {
		IndexedMatWrapper imw = _preConversionQueue->get();
		if(imw.isPoison()) {
			_postConversionQueue->add(imw);
			break;
		}

		MatWrapper mw = imw.matWrapper;
		mw = _converter->convert(mw);
		imw.matWrapper = mw;
		_postConversionQueue->add(imw);
	}
}

}}}
