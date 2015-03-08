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
	int framesCount =  videoReader.get(CV_CAP_PROP_FRAME_COUNT);

	clock_t videoBegin = clock();

	VideoReaderThread reader(&videoReader, _preConversionQueues, _threadNumber,
			converter->getService());
	VideoWriterThread writer(&videoWriter, _postConversionQueues, _threadNumber,
			converter->getService());

	boost::thread readerThread(reader);
	boost::thread writerThread(writer);

	for(int i = 0; i < _threadNumber; i++) {
		ConversionThread conversion(converter, _preConversionQueues[i],
				_postConversionQueues[i]);
		boost::thread conversionThread(conversion);
	}

	writerThread.join();

	videoReader.release();
	videoWriter.release();

	clock_t videoEnd = clock();
	double videoTime = videoEnd - videoBegin;

	std::cout << "video time: " << videoTime / _threadNumber / CLOCKS_PER_SEC << "s, frames count: " << framesCount << std::endl;
	std::cout << "avg frame time: " << int(videoTime / framesCount * 1000 / CLOCKS_PER_SEC / _threadNumber) << "ms" << std::endl;
}

}}}
