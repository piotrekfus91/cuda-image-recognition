#ifndef VIDEOCONVERTER_H_
#define VIDEOCONVERTER_H_

#include "cir/common/MatWrapper.h"
#include "cir/common/ImageProcessingService.h"

namespace cir { namespace common { namespace video {

class VideoConverter {
public:
	VideoConverter(cir::common::ImageProcessingService* service);
	virtual ~VideoConverter();

	virtual cir::common::MatWrapper convert(cir::common::MatWrapper& input) = 0;

	virtual cir::common::ImageProcessingService* getService();

protected:
	cir::common::ImageProcessingService* _service;
};

}}}
#endif /* VIDEOCONVERTER_H_ */
