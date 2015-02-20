#ifndef EUCLIDPATTERNHEURISTIC_H_
#define EUCLIDPATTERNHEURISTIC_H_

#include "cir/common/recognition/heuristic/PatternHeuristic.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

class EuclidPatternHeuristic : public PatternHeuristic {
public:
	EuclidPatternHeuristic();
	virtual ~EuclidPatternHeuristic();

	virtual bool isBetter(double previous, double current) const;
	virtual bool isApplicable(double value) const;
	virtual bool shouldNormalize() const;
	virtual double getFirstValue() const;

protected:
	virtual const double doCountHeuristic(double* huMoments, double* patternHuMoments) const;
};

}}}}
#endif /* EUCLIDPATTERNHEURISTIC_H_ */
