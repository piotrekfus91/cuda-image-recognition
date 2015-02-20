#include <cmath>
#include "cir/common/recognition/heuristic/EuclidPatternHeuristic.h"
#include "cir/common/config.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

EuclidPatternHeuristic::EuclidPatternHeuristic() {

}

EuclidPatternHeuristic::~EuclidPatternHeuristic() {

}

const double EuclidPatternHeuristic::doCountHeuristic(double* huMoments, double* patternHuMoments) const {
	double sum = .0;
	for(int i = 0; i < HU_MOMENTS_NUMBER; i++) {
		double patternHuMoment = patternHuMoments[i];
		double huMoment = huMoments[i];
		double diff = patternHuMoment - huMoment;
		sum += diff * diff;
	}
	return sqrt(sum);
}

bool EuclidPatternHeuristic::isBetter(double previous, double current) const {
	return current < previous;
}

bool EuclidPatternHeuristic::isApplicable(double value) const {
	return true;
}

bool EuclidPatternHeuristic::shouldNormalize() const {
	return true;
}

double EuclidPatternHeuristic::getFirstValue() const {
	return 10000000.;
}

}}}}
