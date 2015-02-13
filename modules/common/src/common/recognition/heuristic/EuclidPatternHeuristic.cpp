#include <cmath>
#include "cir/common/recognition/heuristic/EuclidPatternHeuristic.h"
#include "cir/common/config.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

EuclidPatternHeuristic::EuclidPatternHeuristic() {

}

EuclidPatternHeuristic::~EuclidPatternHeuristic() {

}

const double EuclidPatternHeuristic::countHeuristic(Pattern* pattern, double* huMoments, int segmentIndex) const {
	double sum = .0;
	for(int i = 0; i < HU_MOMENTS_NUMBER; i++) {
		double patternHuMoment = pattern->getHuMoment(segmentIndex, i);
		double huMoment = huMoments[i];
		double diff = patternHuMoment - huMoment;
		sum += diff * diff;
	}
	return sqrt(sum);
}

}}}}
