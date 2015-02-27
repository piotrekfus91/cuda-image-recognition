#include "cir/common/exception/VideoException.h"

namespace cir { namespace common { namespace exception {

VideoException::VideoException(std::string msg) : _msg(msg) {

}

VideoException::~VideoException() throw() {

}

const char* VideoException::what() {
	return _msg.c_str();
}

}}}
