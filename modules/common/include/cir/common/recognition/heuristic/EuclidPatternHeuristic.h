#ifndef EUCLIDPATTERNHEURISTIC_H_
#define EUCLIDPATTERNHEURISTIC_H_

#include "cir/common/recognition/heuristic/PatternHeuristic.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

class EuclidPatternHeuristic : public PatternHeuristic {
public:
	EuclidPatternHeuristic();
	virtual ~EuclidPatternHeuristic();

	virtual const double countHeuristic(cir::common::recognition::Pattern* pattern,
			double* huMoments, int segmentIndex = 0) const;
};

}}}}
#endif /* EUCLIDPATTERNHEURISTIC_H_ */
