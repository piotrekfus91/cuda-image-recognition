#ifndef COSINEPATTERNHEURISTIC_H_
#define COSINEPATTERNHEURISTIC_H_

#include "cir/common/recognition/heuristic/PatternHeuristic.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

class CosinePatternHeuristic : public PatternHeuristic {
public:
	CosinePatternHeuristic();
	virtual ~CosinePatternHeuristic();

	virtual bool isBetter(double previous, double current) const;
	virtual bool isApplicable(double value) const;
	virtual bool shouldNormalize() const;
	virtual double getFirstValue() const;

protected:
	virtual const double doCountHeuristic(double* huMoments, double* patternHuMoments) const;
};

}}}}
#endif /* COSINEPATTERNHEURISTIC_H_ */
