#include "cir/common/classification/HuMomentsClassifier.h"
#include "cir/common/recognition/heuristic/CosinePatternHeuristic.h"
#include "cir/common/recognition/heuristic/EuclidPatternHeuristic.h"
#include "cir/common/recognition/heuristic/SimpleRatioPatternHeuristic.h"

using namespace std;
using namespace cir::common;
using namespace cir::common::recognition;
using namespace cir::common::recognition::heuristic;

namespace cir { namespace common { namespace classification {

HuMomentsClassifier::HuMomentsClassifier()
		: _heuristic(new EuclidPatternHeuristic) {

}

HuMomentsClassifier::~HuMomentsClassifier() {

}

void HuMomentsClassifier::setHeuristic(PatternHeuristic* heuristic) {
	_heuristic = heuristic;
}

bool HuMomentsClassifier::singleChar() const {
	return true;
}

string HuMomentsClassifier::detect(MatWrapper& input, ImageProcessingService* service,
				const map<string, Pattern*>* patternsMap, Segment* segment) {
	double* huMoments = service->countHuMoments(input);

	double bestHeuristic = _heuristic->getFirstValue();
	string bestMatching;

	for(map<string, Pattern*>::const_iterator it = patternsMap->begin(); it != patternsMap->end(); it++) {
		double currentHeuristic = _heuristic->countHeuristic(it->second, huMoments);
		if(_heuristic->isBetter(bestHeuristic, currentHeuristic)) {
			bestMatching = it->first;
			bestHeuristic = currentHeuristic;
		}
	}

	if(_heuristic->isApplicable(bestHeuristic)) {
		return bestMatching;
	} else {
		return "";
	}
}

}}}
