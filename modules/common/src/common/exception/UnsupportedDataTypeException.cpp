#include "cir/common/exception/UnsupportedDataTypeException.h"

namespace cir { namespace common { namespace exception {

std::string UnsupportedDataTypeException::MSG = "MatWrapper currently supports only CV_8U* types of Mat/GpuMat";

const char* UnsupportedDataTypeException::what() {
	return MSG.c_str();
}

}}}
