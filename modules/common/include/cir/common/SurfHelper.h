#ifndef SURFHELPER_H_
#define SURFHELPER_H_

#include "cir/common/ImageProcessingService.h"
#include <vector>

namespace cir { namespace common {

class SurfHelper {
public:
	SurfHelper(ImageProcessingService* service);
	virtual ~SurfHelper();

	std::vector<std::pair<Segment*, int> > findBestPairs(SurfPoints& oldSurfPoints,
			SurfPoints& currentSurfPoints, std::vector<std::pair<Segment*, int> >& oldPairs,
			SegmentArray* currentSegmentArray);
	std::vector<std::pair<Segment*, int> > generateFirstPairs(SegmentArray* segmentArray);

private:
	ImageProcessingService* _service;
	int _counter;
};

}}
#endif /* SURFHELPER_H_ */
