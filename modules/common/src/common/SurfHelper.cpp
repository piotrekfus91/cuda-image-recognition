#include "cir/common/SurfHelper.h"
#include "cir/common/config.h"

namespace cir { namespace common {

SurfHelper::SurfHelper(ImageProcessingService* service) : _service(service), _counter(0) {

}

SurfHelper::~SurfHelper() {

}

std::vector<std::pair<Segment*, int> > SurfHelper::findBestPairs(SurfPoints& oldSurfPoints,
		SurfPoints& currentSurfPoints, std::vector<std::pair<Segment*, int> >& oldPairs,
		SegmentArray* currentSegmentArray) {
	SurfApi* surfApi = _service->getSurfApi();

	std::vector<cv::DMatch> matches = surfApi->findMatches(oldSurfPoints, currentSurfPoints);
	std::vector<std::pair<Segment*, int> > newPairs;

	for(int segm1idx = 0; segm1idx < currentSegmentArray->size; segm1idx++) {
		Segment* currentSegm = currentSegmentArray->segments[segm1idx];

		float bestMatch = 10000;
		int bestId = -1;
		std::vector<std::pair<Segment*, int> >::iterator bestIt;

		for(std::vector<std::pair<Segment*, int> >::iterator it = oldPairs.begin();
				it != oldPairs.end(); it++) {
			std::pair<Segment*, int> oldPair = *it;
			Segment* oldSegm = oldPair.first;
			int oldId = oldPair.second;

			float currentMatch = surfApi->getSimilarity(currentSurfPoints, currentSegm,
					oldSurfPoints, oldSegm, matches);
			if(currentMatch < bestMatch && currentMatch < SURF_THRESHOLD) {
				bestId = oldId;
				bestMatch = currentMatch;
				bestIt = it;
			}
		}

		if(bestId != -1) {
			oldPairs.erase(bestIt);
			newPairs.push_back(std::pair<Segment*, int>(copySegment(currentSegm), bestId));
		} else {
			newPairs.push_back(std::pair<Segment*, int>(copySegment(currentSegm), _counter++));
		}
	}

	return newPairs;
}

std::vector<std::pair<Segment*, int> > SurfHelper::generateFirstPairs(SegmentArray* segmentArray) {
	std::vector<std::pair<Segment*, int> > pairs;
	for(int i = 0; i < segmentArray->size; i++) {
		Segment* segment = segmentArray->segments[i];
		std::pair<Segment*, int> pair(copySegment(segment), _counter++);
		pairs.push_back(pair);
	}
	return pairs;
}

}}
