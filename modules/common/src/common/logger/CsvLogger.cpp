#include "cir/common/logger/CsvLogger.h"
#include <iostream>

namespace cir { namespace common { namespace logger {

char CsvLogger::SEPARATOR = ';';

CsvLogger::CsvLogger() {

}

CsvLogger::~CsvLogger() {

}

void CsvLogger::log(const char* str) {
	_buffer << _module << SEPARATOR << str << std::endl;
}

void CsvLogger::log(const char* str, double d) {
	_buffer << _module << SEPARATOR << str << SEPARATOR << d << std::endl;
}

}}}
