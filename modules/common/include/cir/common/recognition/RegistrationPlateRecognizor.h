#ifndef REGISTRATIONPLATERECOGNIZOR_H_
#define REGISTRATIONPLATERECOGNIZOR_H_

#include "cir/common/ImageProcessingService.h"
#include "cir/common/recognition/Pattern.h"
#include "cir/common/recognition/Recognizor.h"
#include "cir/common/recognition/heuristic/PatternHeuristic.h"
#include <map>

namespace cir { namespace common { namespace recognition {

class RegistrationPlateRecognizor : Recognizor {
public:
	RegistrationPlateRecognizor(cir::common::ImageProcessingService& service);
	virtual ~RegistrationPlateRecognizor();

	virtual const RecognitionInfo recognize(cir::common::MatWrapper& input) const;
	virtual void learn(cir::common::MatWrapper& input);
	virtual void learn(const char* filePath);

	void setPatternHeuristic(cir::common::recognition::heuristic::PatternHeuristic* patternHeuristic);

private:
	cir::common::MatWrapper detectAllColors(cir::common::MatWrapper& input) const;
	cir::common::MatWrapper detectBlue(cir::common::MatWrapper& input) const;
	cir::common::MatWrapper detectWhite(cir::common::MatWrapper& input) const;

	std::map<std::string, Pattern*> _patternsMap;
	cir::common::recognition::heuristic::PatternHeuristic* _patternHeuristic;
};

}}}
#endif /* REGISTRATIONPLATERECOGNIZOR_H_ */
