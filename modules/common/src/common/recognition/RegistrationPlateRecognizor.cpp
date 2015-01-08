#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "opencv2/opencv.hpp"

using namespace cir::common;

namespace cir { namespace common { namespace recognition {

RegistrationPlateRecognizor::RegistrationPlateRecognizor(ImageProcessingService& service) : Recognizor(service) {

}

RegistrationPlateRecognizor::~RegistrationPlateRecognizor() {

}

const RecognitionInfo RegistrationPlateRecognizor::recognize(MatWrapper& input) const {
	_service.init(input.getWidth(), input.getHeight());
	cv::namedWindow("orig");
	cv::imshow("orig", input.getMat());
	cv::waitKey(0);
	MatWrapper mw = _service.bgrToHsv(input);
//	mw = _service.lowPass(mw);
	mw = detectAllColors(mw);
	SegmentArray* allSegmentsArray = _service.segmentate(mw);
	mw = _service.hsvToBgr(mw);

	for(int i = 0; i < allSegmentsArray->size; i++) {
		Segment* segment = allSegmentsArray->segments[i];
		MatWrapper segmentMw = _service.crop(mw, segment);
		cv::namedWindow("seg");
		cv::imshow("seg", segmentMw.getMat());
		cv::waitKey(0);
		segmentMw = _service.bgrToHsv(segmentMw);

		MatWrapper blueMw = detectBlue(segmentMw);
		SegmentArray* blueSegmentsArray = _service.segmentate(blueMw);
		Segment* blueSegment = NULL;
		blueMw = _service.hsvToBgr(blueMw);
		blueMw = _service.mark(blueMw, blueSegmentsArray);
		for(int j = 0; j < blueSegmentsArray->size; j++) {
			Segment* candidate = blueSegmentsArray->segments[j];
			if(candidate->leftX < 3) {
				blueSegment = candidate;
				break;
			}
		}

		if(blueSegment == NULL) {
			return RecognitionInfo(false, NULL);
		}

		MatWrapper whiteMw = detectWhite(segmentMw);
		SegmentArray* whiteSegmentsArray = _service.segmentate(whiteMw);
		segmentMw = _service.mark(whiteMw, whiteSegmentsArray);

		MatWrapper whitePlate;
		bool whitePlateFound = false;

		for(int j = 0; j < whiteSegmentsArray->size; j++) {
			Segment* candidate = whiteSegmentsArray->segments[j];
			if(candidate->rightX > segmentMw.getWidth() - segmentMw.getWidth() * 0.05) {
				whitePlate = _service.crop(whiteMw, candidate);
				whitePlateFound = true;
				break;
			}
		}

		if(whitePlateFound) {
			whitePlate = _service.hsvToBgr(whitePlate);
			cv::namedWindow("white");
			cv::imshow("white", whitePlate.getMat());
			cv::waitKey(0);
		}
	}

	RecognitionInfo recognitionInfo(false, NULL);
	return recognitionInfo;
}

void RegistrationPlateRecognizor::learn(cir::common::MatWrapper& input) {

}

void RegistrationPlateRecognizor::learn(const char* filePath) {

}

MatWrapper RegistrationPlateRecognizor::detectAllColors(MatWrapper& input) const {
	HsvRange ranges[2] = {getBlueRange(), getWhiteRange()};

	return _service.detectColorHsv(input, 2, ranges);
}

MatWrapper RegistrationPlateRecognizor::detectBlue(MatWrapper& input) const {
	HsvRange ranges[1] = {getBlueRange()};

	return _service.detectColorHsv(input, 1, ranges);
}

MatWrapper RegistrationPlateRecognizor::detectWhite(MatWrapper& input) const {
	HsvRange ranges[1] = {getWhiteRange()};

	return _service.detectColorHsv(input, 1, ranges);
}

}}}
