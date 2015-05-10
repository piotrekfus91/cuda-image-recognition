#include "cir/common/config.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "cir/common/recognition/heuristic/EuclidPatternHeuristic.h"
#include "cir/common/recognition/heuristic/SimpleRatioPatternHeuristic.h"
#include "cir/common/recognition/heuristic/CosinePatternHeuristic.h"
#include "cir/common/classification/TesseractClassifier.h"
#include "cir/common/classification/HuMomentsClassifier.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <list>
#include <iostream>

using namespace cir::common;
using namespace cir::common::classification;
using namespace cir::common::recognition::heuristic;
using namespace std;

namespace cir { namespace common { namespace recognition {

RegistrationPlateRecognizor::RegistrationPlateRecognizor(ImageProcessingService& service)
		: Recognizor(service), _classifier(new TesseractClassifier), _writeLetters(false) {

}

RegistrationPlateRecognizor::~RegistrationPlateRecognizor() {

}

void RegistrationPlateRecognizor::setWriteLetters(bool writeLetters) {
	_writeLetters = writeLetters;
}

void RegistrationPlateRecognizor::setClassifier(Classifier* classifier) {
	_classifier = classifier;
}

const RecognitionInfo RegistrationPlateRecognizor::doRecognize(MatWrapper& input) {
	std::list<std::string> recognizedPlates;
	std::list<Segment*> recognizedSegments;

	MatWrapper mw = _service.bgrToHsv(input);
	mw = detectAllColors(mw);
	SegmentArray* allSegmentsArray = _service.segmentate(mw);

	for(int i = 0; i < allSegmentsArray->size; i++) {
		Segment* segment = allSegmentsArray->segments[i];
		if(segment->rightX - segment->leftX == input.getWidth() - 1)
			continue;

		MatWrapper segmentMw = _service.crop(mw, segment);

		MatWrapper blueMw = detectBlue(segmentMw);
		SegmentArray* blueSegmentsArray = _service.segmentate(blueMw);
		Segment* blueSegment = NULL;
		if(blueSegmentsArray->size > 2) {
			release(blueSegmentsArray);
			continue;
		}

		int segmentWidth = segment->rightX - segment->leftX;
		for(int j = 0; j < blueSegmentsArray->size; j++) {
			Segment* candidate = blueSegmentsArray->segments[j];

			if(candidate->leftX <= 0.05 * segmentWidth && candidate->rightX <= 0.2 * segmentWidth) {
				blueSegment = candidate;
				break;
			}
		}

		if(blueSegment == NULL) {
			release(blueSegmentsArray);
			continue;
		}

		release(blueSegmentsArray);

		MatWrapper whiteMw = detectWhite(segmentMw);
		SegmentArray* whiteSegmentsArray = _service.segmentate(whiteMw);

		for(int j = 0; j < whiteSegmentsArray->size; j++) {
			Segment* candidate = whiteSegmentsArray->segments[j];
			if(candidate->rightX - candidate->leftX > 0.75 * (segment->rightX - segment->leftX)) {
				MatWrapper whitePlate = _service.crop(whiteMw, candidate);
				whitePlate = _service.toGrey(whitePlate);
				whitePlate = _service.threshold(whitePlate, true);
				whitePlate = _service.median(whitePlate);

				TesseractClassifier classifier;

				if(classifier.singleChar()) {
					SegmentArray* signsArray = _service.segmentate(whitePlate);

					if(signsArray->size > 3) {
						std::string result = "";
						for(int k = 0; k < signsArray->size; k++) {
							Segment* signSegment = signsArray->segments[k];
							std::string recognized = classifier.detect(whitePlate, &_service, &_patternsMap, signSegment);
							result.append(recognized);
						}

						if(result.size() > 2) {
							if(std::find(recognizedPlates.begin(), recognizedPlates.end(), result) == recognizedPlates.end()) {
								recognizedPlates.push_back(result);
								Segment* resultSegment = copySegment(candidate);
								resultSegment->leftX += segment->leftX;
								resultSegment->rightX += segment->leftX;
								resultSegment->topY += segment->topY;
								resultSegment->bottomY += segment->topY;
								recognizedSegments.push_back(resultSegment);
							}
						}
						break;
					}
				} else {
					std::string result = _classifier->detect(whitePlate, &_service, &_patternsMap, NULL);
					if(result.size() > 2) {
						if(std::find(recognizedPlates.begin(), recognizedPlates.end(), result) == recognizedPlates.end()) {
							recognizedPlates.push_back(result);
							Segment* resultSegment = copySegment(candidate);
							resultSegment->leftX += segment->leftX;
							resultSegment->rightX += segment->leftX;
							resultSegment->topY += segment->topY;
							resultSegment->bottomY += segment->topY;
							recognizedSegments.push_back(resultSegment);
						}
					}
				}
			}
		}
	}

	release(allSegmentsArray);

	if(_writeLetters) {
		for(std::list<std::string>::iterator it = recognizedPlates.begin(); it != recognizedPlates.end(); it++) {
			std::cout << *it << std::endl;
		}
	}

	SegmentArray* resultSegmentArray = NULL;
	bool recognized = false;
	if(recognizedSegments.size() > 0) {
		recognized = true;
		resultSegmentArray = (SegmentArray*) malloc(sizeof(SegmentArray));
		resultSegmentArray->size = recognizedSegments.size();
		resultSegmentArray->segments = (Segment**) malloc(sizeof(Segment*) * resultSegmentArray->size);
		int index = 0;
		for(std::list<Segment*>::iterator it = recognizedSegments.begin(); it != recognizedSegments.end(); it++) {
			resultSegmentArray->segments[index] = *it;
			index++;
		}
	}

	RecognitionInfo recognitionInfo(recognized, resultSegmentArray);
	return recognitionInfo;
}

void RegistrationPlateRecognizor::learn(cir::common::MatWrapper& input) {

}

void RegistrationPlateRecognizor::learn(const char* filePath) {
//	string filePathStr(filePath);
//	unsigned int lastDirSeparatorPos = filePathStr.find_last_of(PATH_SEPARATOR);
//	string fileName;
//	if(lastDirSeparatorPos != string::npos) {
//		fileName = filePathStr.substr(lastDirSeparatorPos + 1);
//	} else {
//		fileName = filePath;
//	}
//
//	int extensionStart = fileName.find_last_of(".");
//	string fileNameCore = fileName.substr(0, extensionStart);
//
//	cv::Mat mat = cv::imread(filePath);
//	MatWrapper mw = _service.getMatWrapper(mat);
//
//	mw = _service.toGrey(mw);
//	mw = _service.threshold(mw, REGISTRATION_PLATE_PATTERN_INVERTED, 127);
//
//	double** huMoments = new double*[1];
//	huMoments[0] = _service.countHuMoments(mw);
//	Pattern* pattern = new Pattern(filePath, 1, huMoments);
//
//	_patternsMap[fileNameCore] = pattern;
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
