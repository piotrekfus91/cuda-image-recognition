#include "cir/common/logger/BufferedConfigurableLogger.h"
#include <string.h>
#include <boost/lexical_cast.hpp>

namespace cir { namespace common { namespace logger {

BufferedConfigurableLogger::BufferedConfigurableLogger(std::list<std::string>& printableLogs) {
	_printableLogs = printableLogs;
}

BufferedConfigurableLogger::~BufferedConfigurableLogger() {

}

Logger* BufferedConfigurableLogger::clone() {
	BufferedConfigurableLogger* copy = new BufferedConfigurableLogger(_printableLogs);
	copy->setModule(_module.c_str());
	return copy;
}

void BufferedConfigurableLogger::log(const char* str) {
	for(std::list<std::string>::iterator it = _printableLogs.begin(); it != _printableLogs.end(); it++) {
		if(*it == str) {
			_buffer.append(_module);
			_buffer.append(": ");
			_buffer.append(str);
			_buffer.append("\n");
		}
	}
}

void BufferedConfigurableLogger::log(const char* str, double d) {
	for(std::list<std::string>::iterator it = _printableLogs.begin(); it != _printableLogs.end(); it++) {
		if(*it == str) {
			_buffer.append(_module);
			_buffer.append(": ");
			_buffer.append(str);
			_buffer.append(": ");
			_buffer.append(boost::lexical_cast<std::string>(d));
			_buffer.append("\n");
		}
	}
}
void BufferedConfigurableLogger::flushBuffer() {
	std::cout << _buffer << std::endl;
	_buffer = "";
}

}}}
