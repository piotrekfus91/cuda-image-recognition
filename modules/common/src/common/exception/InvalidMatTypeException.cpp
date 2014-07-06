#include "common/exception/InvalidMatTypeException.h"

using namespace std;

using namespace cir::common::exception;

string InvalidMatTypeException::MSG = "MatWrapper currently holds another type of Mat/GpuMat";

const char* InvalidMatTypeException::what() {
	return MSG.c_str();
}
