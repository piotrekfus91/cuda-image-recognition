#include "cir/common/video/RecognitionVideoConverter.h"
#include "cir/common/config.h"

using namespace cir::common;
using namespace cir::common::recognition;

namespace cir { namespace common { namespace video {

RecognitionVideoConverter::RecognitionVideoConverter(Recognizor* recognizor, ImageProcessingService* service)
		: VideoConverter(service), _recognizor(recognizor), _surfHelper(service), _withSurf(false) {

}

RecognitionVideoConverter::~RecognitionVideoConverter() {

}

void RecognitionVideoConverter::withSurf() {
	_withSurf = true;
}

void RecognitionVideoConverter::withoutSurf() {
	_withSurf = false;
}

MatWrapper RecognitionVideoConverter::convert(MatWrapper input) {
	RecognitionInfo recognitionInfo = _recognizor->recognize(input);
	if(recognitionInfo.isSuccess()) {
		if(_withSurf) {
			MatWrapper greyInput = _service->toGrey(input);
			if(!_recentPairs.empty()) {
				SurfPoints currentSurfPoints = _service->getSurfApi()->performSurf(greyInput, SURF_MIN_HESSIAN);
				_recentPairs = _surfHelper.findBestPairs(_recentSurfPoints, currentSurfPoints, _recentPairs,
						recognitionInfo.getMatchedSegments());
				_recentSurfPoints = currentSurfPoints;
			} else {
				_recentPairs = _surfHelper.generateFirstPairs(recognitionInfo.getMatchedSegments());
				_recentSurfPoints = _service->getSurfApi()->performSurf(greyInput, SURF_MIN_HESSIAN);
			}
			return _service->mark(input, _recentPairs);
		} else {
			return _service->mark(input, recognitionInfo.getMatchedSegments());
		}
	} else {
		return input;
	}
}

}}}
