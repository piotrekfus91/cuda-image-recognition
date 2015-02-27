#ifndef HUMOMENTSCLASSIFIER_H_
#define HUMOMENTSCLASSIFIER_H_

#include "cir/common/classification/Classifier.h"
#include "cir/common/recognition/heuristic/PatternHeuristic.h"

namespace cir { namespace common { namespace classification {

class HuMomentsClassifier : public Classifier {
public:
	HuMomentsClassifier();
	virtual ~HuMomentsClassifier();

	virtual std::string detect(cir::common::MatWrapper& input, cir::common::ImageProcessingService* service,
				const std::map<std::string, cir::common::recognition::Pattern*>* patternsMap,
				cir::common::Segment* segment = NULL);

	virtual void setHeuristic(cir::common::recognition::heuristic::PatternHeuristic* heuristic);

	virtual bool singleChar() const;

protected:
	cir::common::recognition::heuristic::PatternHeuristic* _heuristic;
};

}}}
#endif /* HUMOMENTSCLASSIFIER_H_ */
