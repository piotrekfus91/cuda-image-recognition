#include "cir/common/test_file_loader.h"
#include "cir/common/recognition/MetroRecognizor.h"
#include "cir/common/recognition/heuristic/EuclidPatternHeuristic.h"
#include "cir/common/recognition/heuristic/SimpleRatioPatternHeuristic.h"
#include "cir/common/recognition/heuristic/CosinePatternHeuristic.h"
#include <list>
#include <opencv2/opencv.hpp>

using namespace cir::common;
using namespace cir::common::recognition::heuristic;

namespace cir { namespace common { namespace recognition {

MetroRecognizor::MetroRecognizor(ImageProcessingService& service) : Recognizor(service), _heuristic(new EuclidPatternHeuristic) {

}

MetroRecognizor::~MetroRecognizor() {

}

const RecognitionInfo MetroRecognizor::recognize(MatWrapper& input) {
	MatWrapper inputHsv = _service.bgrToHsv(input);
	MatWrapper mw = detectAllColors(inputHsv);
	mw = _service.toGrey(mw);
	mw = _service.median(mw);
	mw = _service.threshold(mw);

	SegmentArray* segmentArray = _service.segmentate(mw);
	std::list<Segment*> acceptedSegments;

	for(int i = 0; i < segmentArray->size; i++) {
		Segment* segment = segmentArray->segments[i];
		MatWrapper segmentMw = _service.crop(inputHsv, segment);

		MatWrapper redSegmentMw = detectRed(segmentMw);
		redSegmentMw = _service.toGrey(redSegmentMw);
		redSegmentMw = _service.threshold(redSegmentMw);

		double* redHuMoments = _service.countHuMoments(redSegmentMw);
		double redResult = _heuristic->countHeuristic(&_pattern, redHuMoments, 0);
		if(_heuristic->isApplicable(redResult)) {
			MatWrapper yellowSegmentMw = detectYellow(segmentMw);
			yellowSegmentMw = _service.toGrey(yellowSegmentMw);
			yellowSegmentMw = _service.threshold(yellowSegmentMw);

			double* yellowHuMoments = _service.countHuMoments(yellowSegmentMw);
			double yellowResult = _heuristic->countHeuristic(&_pattern, yellowHuMoments, 1);
			if(_heuristic->isApplicable(yellowResult)) {
				acceptedSegments.push_back(segment);
			}
		}
	}

	if(acceptedSegments.size() > 0) {
		SegmentArray* segmentArray = (SegmentArray*) malloc(sizeof(SegmentArray));
		Segment** segments = (Segment**) malloc(sizeof(Segment*) * acceptedSegments.size());

		int i = 0;
		for(std::list<Segment*>::iterator it = acceptedSegments.begin(); it != acceptedSegments.end(); it++) {
			Segment* s = copySegment(*it);
			segments[i] = s;
			i++;
		}

		segmentArray->size = acceptedSegments.size();
		segmentArray->segments = segments;

		return RecognitionInfo(true, segmentArray);
	} else {
		return RecognitionInfo(false, NULL);
	}
}

void MetroRecognizor::learn(MatWrapper& input) {
	MatWrapper mw = _service.bgrToHsv(input);
	mw = detectRed(mw);
	mw = _service.toGrey(mw);
//	mw = _service.median(mw);
	mw = _service.threshold(mw);
	double* redHuMoments = _service.countHuMoments(mw);

	mw = _service.bgrToHsv(input);
	mw = detectYellow(mw);
	mw = _service.toGrey(mw);
//	mw = _service.median(mw);
	mw = _service.threshold(mw);
	double* yellowHuMoments = _service.countHuMoments(mw);

	double** huMoments = (double**) malloc(sizeof(double*) * 2);
	huMoments[0] = redHuMoments;
	huMoments[1] = yellowHuMoments;

	_pattern.setHuMoments(huMoments);
	_pattern.setSegmentsNumber(2);
}

void MetroRecognizor::learn(const char* filePath) {
	cv::Mat mat = cv::imread(filePath);
	MatWrapper mw = _service.getMatWrapper(mat);
	learn(mw);
	_pattern.setFileName(filePath);
}

MatWrapper MetroRecognizor::detectAllColors(MatWrapper& input) const {
	HsvRange ranges[2] = {getRedRange(), getYellowRange()};

	return _service.detectColorHsv(input, 2, ranges);
}

MatWrapper MetroRecognizor::detectRed(MatWrapper& input) const {
	HsvRange ranges[1] = {getRedRange()};

	return _service.detectColorHsv(input, 1, ranges);
}

MatWrapper MetroRecognizor::detectYellow(MatWrapper& input) const {
	HsvRange ranges[1] = {getYellowRange()};

	return _service.detectColorHsv(input, 1, ranges);
}

}}}
