#include "cir/common/config.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "cir/common/recognition/heuristic/EuclidPatternHeuristic.h"
#include "cir/common/recognition/heuristic/SimpleRatioPatternHeuristic.h"
#include "cir/common/recognition/heuristic/CosinePatternHeuristic.h"
#include "cir/common/classification/TesseractClassifier.h"
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
		: Recognizor(service), _classifier(new TesseractClassifier) {

}

RegistrationPlateRecognizor::~RegistrationPlateRecognizor() {

}

void RegistrationPlateRecognizor::setClassifier(Classifier* classifier) {
	_classifier = classifier;
}

const RecognitionInfo RegistrationPlateRecognizor::recognize(MatWrapper& input) const {
	std::list<std::string> recognizedPlates;
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
		segmentMw = _service.bgrToHsv(segmentMw);

		MatWrapper blueMw = detectBlue(segmentMw);
		SegmentArray* blueSegmentsArray = _service.segmentate(blueMw);
		Segment* blueSegment = NULL;
		blueMw = _service.hsvToBgr(blueMw);
		blueMw = _service.mark(blueMw, blueSegmentsArray);

		int segmentWidth = segment->rightX - segment->leftX;
		for(int j = 0; j < blueSegmentsArray->size; j++) {
			Segment* candidate = blueSegmentsArray->segments[j];

			if(candidate->leftX <= 0.05 * segmentWidth + segment->leftX) {
				blueSegment = candidate;
				break;
			}
		}

		if(blueSegment == NULL) {
			continue;
		}

		MatWrapper whiteMw = detectWhite(segmentMw);
		SegmentArray* whiteSegmentsArray = _service.segmentate(whiteMw);
		segmentMw = _service.mark(whiteMw, whiteSegmentsArray);

		for(int j = 0; j < whiteSegmentsArray->size; j++) {
			Segment* candidate = whiteSegmentsArray->segments[j];
			if(candidate->rightX - candidate->leftX > 0.75 * (segment->rightX - segment->leftX)) {
				MatWrapper whitePlate = _service.crop(whiteMw, candidate);
				whitePlate = _service.toGrey(whitePlate);
				whitePlate = _service.threshold(whitePlate, true);
				cv::namedWindow("white");
				cv::imshow("white", whitePlate.getMat());
				whitePlate = _service.median(whitePlate);
				cv::namedWindow("white2");
				cv::imshow("white2", whitePlate.getMat());

				SegmentArray* signsArray = _service.segmentate(whitePlate);
				cv::namedWindow("sign");
				if(signsArray->size > 3) {
					std::string result = "";
					for(int k = 0; k < signsArray->size; k++) {
						Segment* signSegment = signsArray->segments[k];
						std::string recognized = _classifier->detect(whitePlate, &_service, &_patternsMap, signSegment);
						result.append(recognized);

//						double* huMoments = _service.countHuMoments(signMw);
//						std::string bestSign;
//						double bestHeuristic = 10000000.;
//						for(map<string, Pattern*>::const_iterator it = _patternsMap.begin(); it != _patternsMap.end(); it++) {
//							Pattern* pattern = it->second;
//							const double heuristic = abs(_patternHeuristic->countHeuristic(pattern, huMoments, 0));
//							std::cout << it->first << ": " << heuristic << std::endl;
//							if(bestHeuristic > heuristic) {
//								bestSign = it->first;
//								bestHeuristic = heuristic;
//							}
//						}
					}

					if(result.size() > 0) {
						if(std::find(recognizedPlates.begin(), recognizedPlates.end(), result) == recognizedPlates.end())
							recognizedPlates.push_back(result);
					}
					break;
				}
			}
		}
	}

	for(std::list<std::string>::iterator it = recognizedPlates.begin(); it != recognizedPlates.end(); it++) {
		std::cout << *it << std::endl;
	}

	RecognitionInfo recognitionInfo(false, NULL);
	return recognitionInfo;
}

void RegistrationPlateRecognizor::learn(cir::common::MatWrapper& input) {

}

void RegistrationPlateRecognizor::learn(const char* filePath) {
	string filePathStr(filePath);
	unsigned int lastDirSeparatorPos = filePathStr.find_last_of(PATH_SEPARATOR);
	string fileName;
	if(lastDirSeparatorPos != string::npos) {
		fileName = filePathStr.substr(lastDirSeparatorPos + 1);
	} else {
		fileName = filePath;
	}

	int extensionStart = fileName.find_last_of(".");
	string fileNameCore = fileName.substr(0, extensionStart);

	cv::Mat mat = cv::imread(filePath);
	MatWrapper mw(mat);

	mw = _service.threshold(mw, REGISTRATION_PLATE_PATTERN_INVERTED, 127);

	double** huMoments = new double*[1];
	huMoments[0] = _service.countHuMoments(mw);
	Pattern* pattern = new Pattern(filePath, 1, huMoments);

	_patternsMap[fileNameCore] = pattern;
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
