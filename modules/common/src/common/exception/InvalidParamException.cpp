#include "cir/common/exception/InvalidParamException.h"

using namespace std;

using namespace cir::common::exception;

InvalidParamException::InvalidParamException(string msg) : _msg(msg) {

}

InvalidParamException::~InvalidParamException() throw() {

}

const char* InvalidParamException::what() {
	return _msg.c_str();
}
