#ifndef TESSERACTCLASSIFIER_H_
#define TESSERACTCLASSIFIER_H_

#include <tesseract/baseapi.h>
#include "cir/common/classification/Classifier.h"

namespace cir { namespace common { namespace classification {

class TesseractClassifier: public Classifier {
public:
	TesseractClassifier();
	virtual ~TesseractClassifier();

	virtual std::string detect(cir::common::MatWrapper& input, cir::common::ImageProcessingService* service,
			const std::map<std::string, cir::common::recognition::Pattern*>* patternsMap,
			cir::common::Segment* segment = NULL);

	virtual bool singleChar() const;

private:
	tesseract::TessBaseAPI* _tesseract;
	const int _boundary;
};

}}}
#endif /* TESSERACTCLASSIFIER_H_ */
