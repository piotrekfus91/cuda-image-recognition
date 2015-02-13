#ifndef SIMPLERATIOPATTERNHEURISTIC_H_
#define SIMPLERATIOPATTERNHEURISTIC_H_

#include "cir/common/recognition/heuristic/PatternHeuristic.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

class SimpleRatioPatternHeuristic : public PatternHeuristic {
public:
	SimpleRatioPatternHeuristic();
	virtual ~SimpleRatioPatternHeuristic();

	virtual const double countHeuristic(cir::common::recognition::Pattern* pattern,
			double* huMoments, int segmentIndex = 0) const;
};

}}}}
#endif /* SIMPLERATIOPATTERNHEURISTIC_H_ */
