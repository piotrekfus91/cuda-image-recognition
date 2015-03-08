#ifndef CONVERSIONTHREAD_H_
#define CONVERSIONTHREAD_H_

#include "cir/common/video/VideoConverter.h"
#include "cir/common/concurrency/IndexedMatWrapperBlockingQueue.h"

namespace cir { namespace common { namespace concurrency {

class ConversionThread {
public:
	ConversionThread(cir::common::video::VideoConverter* converter,
			IndexedMatWrapperBlockingQueue* preConversionQueue,
			IndexedMatWrapperBlockingQueue* postConversionQueue);
	virtual ~ConversionThread();

	void operator()();

private:
	cir::common::video::VideoConverter* _converter;
	IndexedMatWrapperBlockingQueue* _preConversionQueue;
	IndexedMatWrapperBlockingQueue* _postConversionQueue;

	bool isPoison(cir::common::MatWrapper& mw);
};

}}}
#endif /* CONVERSIONTHREAD_H_ */
