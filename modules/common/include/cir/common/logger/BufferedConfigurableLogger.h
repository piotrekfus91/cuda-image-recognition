#ifndef BUFFEREDCONFIGURABLELOGGER_H_
#define BUFFEREDCONFIGURABLELOGGER_H_

#include "cir/common/logger/Logger.h"
#include <string>
#include <list>

namespace cir { namespace common { namespace logger {

class BufferedConfigurableLogger : public Logger {
public:
	BufferedConfigurableLogger(std::list<std::string>& printableLogs);
	virtual ~BufferedConfigurableLogger();

	virtual Logger* clone();

	virtual void log(const char* str);
	virtual void log(const char* str, double d);
	virtual void flushBuffer();

private:
	std::list<std::string> _printableLogs;
	std::string _buffer;
};

}}}
#endif /* BUFFEREDCONFIGURABLELOGGER_H_ */
