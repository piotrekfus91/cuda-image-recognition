#include <cmath>
#include <cstdlib>
#include "cir/common/recognition/heuristic/PatternHeuristic.h"
#include "cir/common/config.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

PatternHeuristic::PatternHeuristic() {

}

PatternHeuristic::~PatternHeuristic() {

}

const double PatternHeuristic::countHeuristic(Pattern* pattern,	double* huMoments, int segmentIndex) const {
	double* patternHuMoments = pattern->getHuMoments(segmentIndex);

	if(shouldNormalize()) {
		double* allNormalizedHuMoments = normalize(huMoments, patternHuMoments);
		double* normalizedHuMoments = allNormalizedHuMoments;
		double* normalizedPatternHuMoments = allNormalizedHuMoments + HU_MOMENTS_NUMBER;
		return doCountHeuristic(normalizedHuMoments, normalizedPatternHuMoments);
	} else {
		return doCountHeuristic(huMoments, patternHuMoments);
	}
}

double* PatternHeuristic::normalize(double* huMoments1, double* huMoments2) const {
	double* huMoments = (double*) malloc(sizeof(double) * HU_MOMENTS_NUMBER * 2);

	for(int i = 0; i < HU_MOMENTS_NUMBER; i++) {
		double huMoment1 = huMoments1[i];
		double huMoment2 = huMoments2[i];

		int shift = -log(huMoment1);

		huMoment1 *= pow(10, shift);
		huMoment2 *= pow(10, shift);

		huMoments[i] = huMoment1;
		huMoments[i + HU_MOMENTS_NUMBER] = huMoment2;
	}

	return huMoments;
}

}}}}

