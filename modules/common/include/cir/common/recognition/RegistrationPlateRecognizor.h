#ifndef REGISTRATIONPLATERECOGNIZOR_H_
#define REGISTRATIONPLATERECOGNIZOR_H_

#include "cir/common/ImageProcessingService.h"
#include "cir/common/recognition/Pattern.h"
#include "cir/common/recognition/Recognizor.h"
#include "cir/common/classification/Classifier.h"
#include <map>

namespace cir { namespace common { namespace recognition {

class RegistrationPlateRecognizor : public Recognizor {
public:
	RegistrationPlateRecognizor(cir::common::ImageProcessingService& service);
	virtual ~RegistrationPlateRecognizor();

	virtual const RecognitionInfo recognize(cir::common::MatWrapper& input) const;
	virtual void learn(cir::common::MatWrapper& input);
	virtual void learn(const char* filePath);

	void setClassifier(cir::common::classification::Classifier* classifier);

private:
	cir::common::MatWrapper detectAllColors(cir::common::MatWrapper& input) const;
	cir::common::MatWrapper detectBlue(cir::common::MatWrapper& input) const;
	cir::common::MatWrapper detectWhite(cir::common::MatWrapper& input) const;

	std::map<std::string, Pattern*> _patternsMap;
	cir::common::classification::Classifier* _classifier;
};

}}}
#endif /* REGISTRATIONPLATERECOGNIZOR_H_ */
