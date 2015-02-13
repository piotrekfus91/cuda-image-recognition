#ifndef COSINEPATTERNHEURISTIC_H_
#define COSINEPATTERNHEURISTIC_H_

#include "cir/common/recognition/heuristic/PatternHeuristic.h"

namespace cir { namespace common { namespace recognition { namespace heuristic {

class CosinePatternHeuristic : public PatternHeuristic {
public:
	CosinePatternHeuristic();
	virtual ~CosinePatternHeuristic();

	virtual const double countHeuristic(cir::common::recognition::Pattern* pattern,
				double* huMoments, int segmentIndex = 0) const;
};

}}}}
#endif /* COSINEPATTERNHEURISTIC_H_ */
