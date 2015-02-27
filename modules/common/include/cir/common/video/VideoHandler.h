#ifndef VIDEOHANDLER_H_
#define VIDEOHANDLER_H_

#include <string>
#include "cir/common/video/VideoConverter.h"
#include <opencv2/opencv.hpp>

namespace cir { namespace common { namespace video {

class VideoHandler {
public:
	VideoHandler();
	virtual ~VideoHandler();

	virtual void handle(std::string& inputFilePath, VideoConverter* converter);
	virtual void handle(std::string& inputFilePath, std::string& outputFilePath,
			VideoConverter* converter) = 0;

protected:
	virtual cv::VideoCapture openVideoReader(std::string& inputFilePath) const;
	virtual cv::VideoWriter openVideoWriter(cv::VideoCapture& videoReader, std::string& outputFilePath) const;
};

}}}
#endif /* VIDEOHANDLER_H_ */
