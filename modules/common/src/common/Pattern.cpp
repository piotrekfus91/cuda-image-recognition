#include "cir/common/Pattern.h"
#include <cstdlib>

namespace cir { namespace common {

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

double Pattern::getHuMoment(int segment, int index) {
	return _huMoments[segment][index];
}

}}
