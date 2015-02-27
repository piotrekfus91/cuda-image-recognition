#ifndef SINGLETHREADVIDEOHANDLER_H_
#define SINGLETHREADVIDEOHANDLER_H_

#include "cir/common/video/VideoHandler.h"

namespace cir { namespace common { namespace video {

class SingleThreadVideoHandler : public VideoHandler {
public:
	SingleThreadVideoHandler();
	virtual ~SingleThreadVideoHandler();

	virtual void handle(std::string& inputFilePath, std::string& outputFilePath,
			VideoConverter* converter);
};

}}}
#endif /* SINGLETHREADVIDEOHANDLER_H_ */
