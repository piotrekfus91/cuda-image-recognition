#ifndef RECOGNITIONINFO_H_
#define RECOGNITIONINFO_H_

#include "cir/common/SegmentArray.h"

namespace cir { namespace common { namespace recognition {

class RecognitionInfo {
public:
	RecognitionInfo(const bool success, const SegmentArray* matchedSegments);
	virtual ~RecognitionInfo();

	const bool isSuccess() const;
	const cir::common::SegmentArray* getMatchedSegments() const;

private:
	const bool _success;
	const cir::common::SegmentArray* _matchedSegments;
};

}}}
#endif /* RECOGNITIONINFO_H_ */
