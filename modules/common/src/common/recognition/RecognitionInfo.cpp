#include "cir/common/recognition/RecognitionInfo.h"

using namespace cir::common;

namespace cir { namespace common { namespace recognition {

RecognitionInfo::RecognitionInfo(const bool success, SegmentArray* matchedSegments)
		: _success(success), _matchedSegments(matchedSegments) {

}

RecognitionInfo::~RecognitionInfo() {
	release(_matchedSegments);
}

const bool RecognitionInfo::isSuccess() const {
	return _success;
}

SegmentArray* RecognitionInfo::getMatchedSegments() const {
	return _matchedSegments;
}

}}}
