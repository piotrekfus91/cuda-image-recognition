#include "cir/common/video/RecognitionVideoConverter.h"

using namespace cir::common;
using namespace cir::common::recognition;

namespace cir { namespace common { namespace video {

RecognitionVideoConverter::RecognitionVideoConverter(Recognizor* recognizor, ImageProcessingService* service)
		: VideoConverter(service), _recognizor(recognizor) {

}

RecognitionVideoConverter::~RecognitionVideoConverter() {

}

MatWrapper RecognitionVideoConverter::convert(MatWrapper& input) {
	RecognitionInfo recognitionInfo = _recognizor->recognize(input);
	if(recognitionInfo.isSuccess()) {
		return _service->mark(input, recognitionInfo.getMatchedSegments());
	} else {
		return input;
	}
}

}}}
