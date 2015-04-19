#ifndef RECOGNITIONVIDEOCONVERTER_H_
#define RECOGNITIONVIDEOCONVERTER_H_

#include "cir/common/video/VideoConverter.h"
#include "cir/common/recognition/Recognizor.h"
#include "cir/common/SurfHelper.h"
#include <vector>

namespace cir { namespace common { namespace video {

class RecognitionVideoConverter : public VideoConverter {
public:
	RecognitionVideoConverter(cir::common::recognition::Recognizor* recognizor,
			cir::common::ImageProcessingService* service);
	virtual ~RecognitionVideoConverter();

	virtual cir::common::MatWrapper convert(cir::common::MatWrapper input);

	void withSurf();
	void withoutSurf();

private:
	cir::common::recognition::Recognizor* _recognizor;
	cir::common::SurfPoints _recentSurfPoints;
	std::vector<std::pair<cir::common::Segment*, int> > _recentPairs;
	cir::common::SurfHelper _surfHelper;
	bool _withSurf;
};

}}}
#endif /* RECOGNITIONVIDEOCONVERTER_H_ */
