#include "cir/common/recognition/heuristic/SimpleRatioPatternHeuristic.h"
#include "cir/common/config.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

SimpleRatioPatternHeuristic::SimpleRatioPatternHeuristic() {

}

SimpleRatioPatternHeuristic::~SimpleRatioPatternHeuristic() {

}

const double SimpleRatioPatternHeuristic::countHeuristic(Pattern* pattern, double* huMoments, int segmentIndex) const {
	double sum = .0;
	for(int i = 0; i < HU_MOMENTS_NUMBER; i++) {
		double patternHuMoment = pattern->getHuMoment(segmentIndex, i);
		double huMoment = huMoments[i];
		double ratio = patternHuMoment / huMoment;
		sum += ratio;
	}
	return sum;
}

}}}}
