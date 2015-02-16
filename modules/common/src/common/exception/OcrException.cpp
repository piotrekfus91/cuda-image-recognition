#include "cir/common/exception/OcrException.h"

namespace cir { namespace common { namespace exception {

std::string OcrException::MSG = "OCR exception";

const char* OcrException::what() {
	return MSG.c_str();
}

}}}
