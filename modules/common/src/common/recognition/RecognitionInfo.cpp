#include "cir/common/recognition/RecognitionInfo.h"

using namespace cir::common;

namespace cir { namespace common { namespace recognition {

RecognitionInfo::RecognitionInfo(const bool success, const SegmentArray* matchedSegments)
		: _success(success), _matchedSegments(matchedSegments) {

}

RecognitionInfo::~RecognitionInfo() {

}

const bool RecognitionInfo::isSuccess() const {
	return _success;
}

const SegmentArray* RecognitionInfo::getMatchedSegments() const {
	return _matchedSegments;
}

}}}
