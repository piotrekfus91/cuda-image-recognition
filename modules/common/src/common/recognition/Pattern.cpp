#include "cir/common/recognition/Pattern.h"
#include "cir/common/config.h"
#include <cstdlib>
#include <iostream>
#include <cmath>

namespace cir { namespace common { namespace recognition {

Pattern::Pattern() : _fileName(""), _segmentsNumber(0), _huMoments(NULL) {

}

Pattern::Pattern(std::string fileName, int segmentsNumber, double** huMoments)
	: _fileName(fileName), _segmentsNumber(segmentsNumber), _huMoments(huMoments) {

}

Pattern::Pattern(const Pattern& pattern) {
	_fileName = pattern._fileName;
	_segmentsNumber = pattern._segmentsNumber;
	_huMoments = pattern._huMoments;
}

Pattern::~Pattern() {
	for(int i = 0; i < _segmentsNumber; i++) {
		free(_huMoments[i]);
	}
	free(_huMoments);
}

std::string Pattern::getFileName() {
	return _fileName;
}

void Pattern::setFileName(std::string fileName) {
	_fileName = fileName;
}

double Pattern::getHuMoment(int segment, int index) {
	return _huMoments[segment][index];
}

double* Pattern::getHuMoments(int segment) {
	return _huMoments[segment];
}

void Pattern::setHuMoments(double** huMoments) {
	_huMoments = huMoments;
}

void Pattern::setSegmentsNumber(int segmentsNumber) {
	_segmentsNumber = segmentsNumber;
}

const bool Pattern::matches(int segment, double* huMoments) const {
	int hits = 0;
	for(int i = 0; i < HU_MOMENTS_NUMBER; i++) {
		double ratio;
		double thisMoment = fabs(_huMoments[segment][i]);
		double comingMoment = fabs(huMoments[i]);
		if(thisMoment > comingMoment) {
			ratio = comingMoment / thisMoment;
		} else {
			ratio = thisMoment / comingMoment;
		}

		double minRatio = i == 0 ? 0.6 : 0.05;
		if(ratio >= minRatio)
			hits++;
	}
	return hits > 1;
}

}}}
