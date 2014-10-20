#ifndef RECOGNIZOR_H_
#define RECOGNIZOR_H_

#include "cir/common/ImageProcessingService.h"
#include "cir/common/MatWrapper.h"
#include "cir/common/recognition/RecognitionInfo.h"

namespace cir { namespace common { namespace recognition {

class Recognizor {
public:
	Recognizor(ImageProcessingService& service);
	virtual ~Recognizor();

	virtual const RecognitionInfo recognize(cir::common::MatWrapper& input) const = 0;
	virtual void learn(cir::common::MatWrapper& input) = 0;
	virtual void learn(const char* filePath) = 0;

protected:
	ImageProcessingService& _service;
};

}}}
#endif /* RECOGNIZOR_H_ */
