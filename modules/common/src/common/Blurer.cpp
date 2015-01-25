#include "cir/common/Blurer.h"
#include "cir/common/exception/InvalidParamException.h"
#include "cir/common/exception/InvalidMatTypeException.h"

using namespace cir::common::exception;

namespace cir { namespace common {

Blurer::Blurer() {

}

Blurer::~Blurer() {

}

MatWrapper Blurer::median(const MatWrapper& mw, int size) {
	if(size < 0 || size % 2 != 1)
		throw InvalidParamException("Size must be odd and positive");

	if(mw.getColorScheme() != MatWrapper::GRAY)
		throw InvalidMatTypeException();

	return doMedian(mw, size / 2);
}

}}
