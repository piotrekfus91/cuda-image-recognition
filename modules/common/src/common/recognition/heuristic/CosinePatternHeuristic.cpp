#include <cmath>
#include "cir/common/config.h"
#include "cir/common/recognition/heuristic/CosinePatternHeuristic.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

CosinePatternHeuristic::CosinePatternHeuristic() {

}

CosinePatternHeuristic::~CosinePatternHeuristic() {

}

// http://en.wikipedia.org/wiki/Cosine_similarity
const double CosinePatternHeuristic::doCountHeuristic(double* huMoments, double* patternHuMoments) const {
	double product = .0;
	double huMomentTotal = .0;
	double patternTotal = .0;

	for(int i = 0; i < HU_MOMENTS_NUMBER; i++) {
		double huMoment = huMoments[i];
		double patternHuMoment = patternHuMoments[i];

		product += huMoment * patternHuMoment;
		huMomentTotal += huMoment * huMoment;
		patternTotal += patternHuMoment * patternHuMoment;
	}

	huMomentTotal = sqrt(huMomentTotal);
	patternTotal = sqrt(patternTotal);
	return product / (huMomentTotal * patternTotal);
}

bool CosinePatternHeuristic::isBetter(double previous, double current) const {
	return current > previous;
}

bool CosinePatternHeuristic::isApplicable(double value) const {
	return value > 0.9;
}

bool CosinePatternHeuristic::shouldNormalize() const {
	return true;
}

double CosinePatternHeuristic::getFirstValue() const {
	return 0.;
}

}}}}
