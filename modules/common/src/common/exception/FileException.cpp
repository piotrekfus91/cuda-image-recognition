#include "cir/common/exception/FileException.h"

namespace cir { namespace common { namespace exception {

FileException::FileException(const char* what) : _what(what) {

}

FileException::~FileException() throw() {

}

const char* FileException::what() {
	return _what;
}

}}}
