#include <cmath>
#include "cir/common/config.h"
#include "cir/common/recognition/heuristic/CosinePatternHeuristic.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

CosinePatternHeuristic::CosinePatternHeuristic() {

}

CosinePatternHeuristic::~CosinePatternHeuristic() {

}

// http://en.wikipedia.org/wiki/Cosine_similarity
const double CosinePatternHeuristic::countHeuristic(Pattern* pattern, double* huMoments, int segmentIndex) const {
	double product = .0;
	double huMomentTotal = .0;
	double patternTotal = .0;

	for(int i = 0; i < HU_MOMENTS_NUMBER; i++) {
		double patternHuMoment = pattern->getHuMoment(segmentIndex, i);
		double huMoment = huMoments[i];

		product += huMoment * patternHuMoment;
		huMomentTotal += huMoment * huMoment;
		patternTotal += patternHuMoment * patternHuMoment;
	}

	huMomentTotal = sqrt(huMomentTotal);
	patternTotal = sqrt(patternTotal);
	return acos(product / (huMomentTotal * patternTotal));
}

}}}}
