#include "cir/common/recognition/SpeedLimitRecognizor.h"
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "cir/common/config.h"

using namespace cir::common;

namespace cir { namespace common { namespace recognition {

SpeedLimitRecognizor::SpeedLimitRecognizor(ImageProcessingService& service) : Recognizor(service) {

}

SpeedLimitRecognizor::~SpeedLimitRecognizor() {

}

const RecognitionInfo SpeedLimitRecognizor::recognize(MatWrapper& input) {
	MatWrapper mw = input.clone();
	mw = _service.bgrToHsv(mw);
	MatWrapper inputHsv = mw.clone();
	mw = detectColor(mw);
	SegmentArray* allSegments = _service.segmentate(mw);
	std::vector<Segment> acceptedSegments;

	for(int i = 0; i < allSegments->size; i++) {
		Segment* segment = allSegments->segments[i];
		mw = _service.crop(inputHsv, segment);
		mw = detectColor(mw);
		mw = _service.toGrey(mw);
		if(singleCandidate(mw)) {
			check(acceptedSegments, mw, segment, 0, 0);
		} else {
			mw = _service.erode(mw, 2);
			SegmentArray* smallSegments = _service.segmentate(mw);
			int widthOffset = segment->leftX;
			int heightOffset = segment->bottomY;
			for(int j = 0; j < smallSegments->size; j++) {
				Segment* smallSegment = smallSegments->segments[j];
				MatWrapper smallMw = _service.crop(mw, smallSegment);
				smallMw = _service.dilate(smallMw, 2);
				check(acceptedSegments, smallMw, smallSegment, widthOffset, heightOffset);
			}
		}
	}


	Segment** segments = NULL;
	if(acceptedSegments.size() > 0) {
		segments = (Segment**) malloc(sizeof(Segment*) * acceptedSegments.size());
		for(unsigned int i = 0; i < acceptedSegments.size(); i++) {
			segments[i] = copySegment(&(acceptedSegments[i]));
		}
	}
	SegmentArray* segmentArray = (SegmentArray*) malloc(sizeof(SegmentArray));

	segmentArray->size = acceptedSegments.size();
	segmentArray->segments = segments;
	return RecognitionInfo(acceptedSegments.size() > 0, segmentArray);
}

void SpeedLimitRecognizor::learn(MatWrapper& input) {

}

void SpeedLimitRecognizor::learn(const char* filePath) {
	std::string filePathStr(filePath);
	cv::Mat mat = cv::imread(filePathStr);
	MatWrapper mw = _service.getMatWrapper(mat);
	mw = _service.bgrToHsv(mw);
	mw = detectColor(mw);
	mw = _service.toGrey(mw);
	mw = _service.threshold(mw, 1);
	mw = _service.lowPass(mw);
	cv::namedWindow("pattern");
	cv::imshow("pattern", mw.getMat());
	cv::waitKey(0);
	double* redHuMoments = _service.countHuMoments(mw);

	double** huMoments = (double**) malloc(sizeof(double*) * 1);
	for(int i = 0; i < 1; i++) {
		huMoments[i] = (double*) malloc(sizeof(double) * HU_MOMENTS_NUMBER);
		for(int j = 0; j < HU_MOMENTS_NUMBER; j++) {
			huMoments[i][j] = redHuMoments[j];
		}
	}

	_pattern.setSegmentsNumber(1);
	_pattern.setFileName(filePath);
	_pattern.setHuMoments(huMoments);
}

MatWrapper SpeedLimitRecognizor::detectColor(MatWrapper& input) const {
	HsvRange ranges[1] = {getRedRange()};

	return _service.detectColorHsv(input, 1, ranges);
}

void SpeedLimitRecognizor::check(std::vector<Segment>& acceptedSegments, MatWrapper& input, Segment* segment,
		int widthOffset, int heightOffset) const {
	MatWrapper mw = _service.threshold(input, 1);
	mw = _service.lowPass(mw);
	cv::namedWindow("segment");
	cv::imshow("segment", mw.getMat());
	cv::waitKey(0);
	double* huMoments = _service.countHuMoments(mw);
	if(_pattern.matches(0, huMoments)) {
		segment->leftX += widthOffset;
		segment->rightX += widthOffset;
		segment->topY += heightOffset;
		segment->bottomY += heightOffset;
		acceptedSegments.push_back(*(segment));
	}
}

bool SpeedLimitRecognizor::singleCandidate(MatWrapper& input) const {
	int width = input.getWidth();
	int height = input.getHeight();
	float ratio;
	if(width > height)
		ratio = 1.0 * width / height;
	else
		ratio = 1.0 * height / width;
	return ratio < 1.1;
}

}}}
