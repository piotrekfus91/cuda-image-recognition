#ifndef STREAMHANDLER_H_
#define STREAMHANDLER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#include <boost/thread.hpp>

namespace cir { namespace common { namespace concurrency {

class StreamHandler {
public:
	static cv::gpu::Stream* currentStream();
	static cudaStream_t nativeStream();

private:
	StreamHandler();
	virtual ~StreamHandler();

	static std::map<boost::thread::id, cv::gpu::Stream*> _threadToStreamMap;
};

}}}
#endif /* STREAMHANDLER_H_ */
