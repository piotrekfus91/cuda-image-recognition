#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include "cir/common/MatWrapper.h"
#include "cir/common/ImageProcessingService.h"
#include "cir/common/Segment.h"
#include "cir/common/recognition/Pattern.h"
#include <string>

namespace cir { namespace common { namespace classification {

class Classifier {
public:
	Classifier();
	virtual ~Classifier();

	virtual std::string detect(cir::common::MatWrapper& input, ImageProcessingService* service,
			const std::map<std::string, cir::common::recognition::Pattern*>* patternsMap,
			cir::common::Segment* segment = NULL) = 0;

	virtual bool singleChar() const = 0;
};

}}}
#endif /* CLASSIFIER_H_ */
