#include "cir/common/recognition/SpeedLimitRecognizor.h"

using namespace cir::common;

namespace cir { namespace common { namespace recognition {

SpeedLimitRecognizor::SpeedLimitRecognizor(ImageProcessingService& service) : Recognizor(service) {

}

SpeedLimitRecognizor::~SpeedLimitRecognizor() {

}

const RecognitionInfo SpeedLimitRecognizor::recognize(MatWrapper& input) const {
	return RecognitionInfo(false, NULL);
}

void SpeedLimitRecognizor::learn(MatWrapper& input) {

}

void SpeedLimitRecognizor::learn(const char* filePath) {

}

}}}
