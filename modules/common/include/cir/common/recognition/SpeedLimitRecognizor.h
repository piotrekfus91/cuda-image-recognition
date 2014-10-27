#ifndef SPEEDLIMITRECOGNIZOR_H_
#define SPEEDLIMITRECOGNIZOR_H_

#include "cir/common/recognition/Pattern.h"
#include "cir/common/recognition/Recognizor.h"
#include <vector>

namespace cir { namespace common { namespace recognition {

class SpeedLimitRecognizor : public Recognizor {
public:
	SpeedLimitRecognizor(ImageProcessingService& service);
	virtual ~SpeedLimitRecognizor();

	virtual const RecognitionInfo recognize(cir::common::MatWrapper& input) const;
	virtual void learn(cir::common::MatWrapper& input);
	virtual void learn(const char* filePath);

protected:
	virtual cir::common::MatWrapper detectColor(MatWrapper& input) const;
	Pattern _pattern;

private:
	void check(std::vector<cir::common::Segment>& acceptedSegments, cir::common::MatWrapper& input,
			cir::common::Segment* segment, int widthOffset, int heightOffset) const;
	bool singleCandidate(cir::common::MatWrapper& mw) const;
};

}}}
#endif /* SPEEDLIMITRECOGNIZOR_H_ */
