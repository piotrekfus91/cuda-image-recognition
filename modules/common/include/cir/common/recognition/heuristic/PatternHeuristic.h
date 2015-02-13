#ifndef PATTERNHEURISTIC_H_
#define PATTERNHEURISTIC_H_

#include "cir/common/recognition/Pattern.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

class PatternHeuristic {
public:
	PatternHeuristic();
	virtual ~PatternHeuristic();

	virtual const double countHeuristic(cir::common::recognition::Pattern* pattern,
			double* huMoments, int segmentIndex = 0) const = 0;
};

}}}}
#endif /* PATTERNHEURISTIC_H_ */
