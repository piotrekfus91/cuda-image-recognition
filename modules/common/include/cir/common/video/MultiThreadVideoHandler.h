#ifndef MULTITHREADVIDEOHANDLER_H_
#define MULTITHREADVIDEOHANDLER_H_

#include "cir/common/video/VideoHandler.h"
#include "cir/common/concurrency/IndexedMatWrapperBlockingQueue.h"

namespace cir { namespace common { namespace video {

class MultiThreadVideoHandler : public VideoHandler {
public:
	MultiThreadVideoHandler();
	virtual ~MultiThreadVideoHandler();

	virtual void handle(std::string& inputFilePath, std::string& outputFilePath,
			VideoConverter* converter);

private:
	int _threadNumber;
	cir::common::concurrency::IndexedMatWrapperBlockingQueue** _preConversionQueues;
	cir::common::concurrency::IndexedMatWrapperBlockingQueue** _postConversionQueues;
};

}}}
#endif /* MULTITHREADVIDEOHANDLER_H_ */
