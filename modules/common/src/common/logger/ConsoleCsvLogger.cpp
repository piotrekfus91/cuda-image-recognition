#include "cir/common/logger/ConsoleCsvLogger.h"
#include <iostream>

namespace cir { namespace common { namespace logger {

ConsoleCsvLogger::ConsoleCsvLogger() {

}

ConsoleCsvLogger::~ConsoleCsvLogger() {

}

Logger* ConsoleCsvLogger::clone() {
	ConsoleCsvLogger* logger = new ConsoleCsvLogger();
	logger->setModule(_module.c_str());
	return logger;
}

void ConsoleCsvLogger::flushBuffer() {
	std::cout << _buffer.str();
}

}}}
