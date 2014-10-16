#include "cir/common/exception/InvalidColorSchemeException.h"

namespace cir { namespace common { namespace exception {

std::string InvalidColorSchemeException::MSG = "MatWrapper currently holds another type of Mat/GpuMat";

const char* InvalidColorSchemeException::what() {
	return MSG.c_str();
}

}}}
