#ifndef VIDEOCONVERTER_H_
#define VIDEOCONVERTER_H_

#include "cir/common/MatWrapper.h"

namespace cir { namespace common { namespace video {

class VideoConverter {
public:
	VideoConverter();
	virtual ~VideoConverter();

	virtual cir::common::MatWrapper convert(cir::common::MatWrapper& input) = 0;
};

}}}
#endif /* VIDEOCONVERTER_H_ */
