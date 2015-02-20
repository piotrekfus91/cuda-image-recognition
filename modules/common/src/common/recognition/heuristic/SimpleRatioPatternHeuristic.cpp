#include <cmath>
#include "cir/common/recognition/heuristic/SimpleRatioPatternHeuristic.h"
#include "cir/common/config.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

SimpleRatioPatternHeuristic::SimpleRatioPatternHeuristic() {

}

SimpleRatioPatternHeuristic::~SimpleRatioPatternHeuristic() {

}

const double SimpleRatioPatternHeuristic::doCountHeuristic(double* huMoments, double* patternHuMoments) const {
	double sum = .0;
	for(int i = 0; i < HU_MOMENTS_NUMBER; i++) {
		double patternHuMoment = patternHuMoments[i];
		double huMoment = huMoments[i];
		double ratio = patternHuMoment / huMoment;
		sum += ratio;
	}
	return sum;
}

bool SimpleRatioPatternHeuristic::isBetter(double previous, double current) const {
	double previousAbsDiff = fabs(previous - HU_MOMENTS_NUMBER);
	double currentAbsDiff = fabs(current - HU_MOMENTS_NUMBER);
	return currentAbsDiff < previousAbsDiff;
}

bool SimpleRatioPatternHeuristic::isApplicable(double value) const {
	return true;
}

bool SimpleRatioPatternHeuristic::shouldNormalize() const {
	return false;
}

double SimpleRatioPatternHeuristic::getFirstValue() const {
	return 100000.;
}

}}}}
