#ifndef PATTERNHEURISTIC_H_
#define PATTERNHEURISTIC_H_

#include "cir/common/recognition/Pattern.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

class PatternHeuristic {
public:
	PatternHeuristic();
	virtual ~PatternHeuristic();

public:
	virtual const double countHeuristic(const cir::common::recognition::Pattern* pattern,
					double* huMoments, int segmentIndex = 0) const;
	virtual bool isBetter(double previous, double current) const = 0;
	virtual bool isApplicable(double value) const = 0;
	virtual bool shouldNormalize() const = 0;
	virtual double getFirstValue() const = 0;
	virtual double* normalize(double* huMoments1, double* huMoments2) const;

protected:
	virtual const double doCountHeuristic(double* huMoments, double* patternHuMoments) const = 0;
};

}}}}
#endif /* PATTERNHEURISTIC_H_ */
