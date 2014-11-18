#include "cir/common/logger/NullLogger.h"

namespace cir { namespace common { namespace logger {

NullLogger::NullLogger() {

}

NullLogger::~NullLogger() {

}

Logger* NullLogger::clone() {
	NullLogger* logger = new NullLogger();
	logger->setModule(_module.c_str());
	return logger;
}

void NullLogger::log(const char* str) {

}

void NullLogger::log(const char* str, double d) {

}

void NullLogger::flushBuffer() {
}

}}}
