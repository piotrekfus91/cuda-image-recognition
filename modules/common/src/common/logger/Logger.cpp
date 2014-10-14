#include "cir/common/logger/Logger.h"

namespace cir { namespace common { namespace logger {

Logger::Logger() : _module("") {

}

Logger::~Logger() {

}

void Logger::setModule(const char* module) {
	_module = module;
}

}}}
