#ifndef SIMPLERATIOPATTERNHEURISTIC_H_
#define SIMPLERATIOPATTERNHEURISTIC_H_

#include "cir/common/recognition/heuristic/PatternHeuristic.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

class SimpleRatioPatternHeuristic : public PatternHeuristic {
public:
	SimpleRatioPatternHeuristic();
	virtual ~SimpleRatioPatternHeuristic();

	virtual bool isBetter(double previous, double current) const;
	virtual bool isApplicable(double value) const;
	virtual bool shouldNormalize() const;
	virtual double getFirstValue() const;

protected:
	virtual const double doCountHeuristic(double* huMoments, double* patternHuMoments) const;
};

}}}}
#endif /* SIMPLERATIOPATTERNHEURISTIC_H_ */
