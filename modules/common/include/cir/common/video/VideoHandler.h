#ifndef VIDEOHANDLER_H_
#define VIDEOHANDLER_H_

#include <string>
#include "cir/common/video/VideoConverter.h"

namespace cir { namespace common { namespace video {

class VideoHandler {
public:
	VideoHandler();
	virtual ~VideoHandler();

	virtual void handle(std::string& inputFilePath, VideoConverter* converter);
	virtual void handle(std::string& inputFilePath, std::string& outputFilePath,
			VideoConverter* converter) = 0;
};

}}}
#endif /* VIDEOHANDLER_H_ */
