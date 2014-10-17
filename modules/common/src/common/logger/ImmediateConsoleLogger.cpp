#include "cir/common/logger/ImmediateConsoleLogger.h"
#include <iostream>

namespace cir { namespace common { namespace logger {

ImmediateConsoleLogger::ImmediateConsoleLogger() {

}

ImmediateConsoleLogger::~ImmediateConsoleLogger() {

}

Logger* ImmediateConsoleLogger::clone() {
	ImmediateConsoleLogger* logger = new ImmediateConsoleLogger();
	logger->setModule(_module.c_str());
	return logger;
}

void ImmediateConsoleLogger::log(const char* str) {
	std::cout << _module << ": " << str << std::endl;
}

void ImmediateConsoleLogger::log(const char* str, double d) {
	std::cout << _module << ": " << str << " " << d << std::endl;
}

void ImmediateConsoleLogger::flushBuffer() {

}

}}}
