#ifndef RECOGNITIONVIDEOCONVERTER_H_
#define RECOGNITIONVIDEOCONVERTER_H_

#include "cir/common/video/VideoConverter.h"
#include "cir/common/recognition/Recognizor.h"

namespace cir { namespace common { namespace video {

class RecognitionVideoConverter : public VideoConverter {
public:
	RecognitionVideoConverter(cir::common::recognition::Recognizor* recognizor,
			cir::common::ImageProcessingService* service);
	virtual ~RecognitionVideoConverter();

	virtual cir::common::MatWrapper convert(cir::common::MatWrapper input);

private:
	cir::common::recognition::Recognizor* _recognizor;
};

}}}
#endif /* RECOGNITIONVIDEOCONVERTER_H_ */
