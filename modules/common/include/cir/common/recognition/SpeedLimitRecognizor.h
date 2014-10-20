#ifndef SPEEDLIMITRECOGNIZOR_H_
#define SPEEDLIMITRECOGNIZOR_H_

#include "cir/common/recognition/Recognizor.h"

namespace cir { namespace common { namespace recognition {

class SpeedLimitRecognizor : public Recognizor {
public:
	SpeedLimitRecognizor(ImageProcessingService& service);
	virtual ~SpeedLimitRecognizor();

	virtual const RecognitionInfo recognize(cir::common::MatWrapper& input) const;
	virtual void learn(cir::common::MatWrapper& input);
	virtual void learn(const char* filePath);
};

}}}
#endif /* SPEEDLIMITRECOGNIZOR_H_ */
