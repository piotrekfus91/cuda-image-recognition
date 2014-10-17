#ifndef IMMEDIATECONSOLELOGGER_H_
#define IMMEDIATECONSOLELOGGER_H_

#include "cir/common/logger/Logger.h"

namespace cir { namespace common { namespace logger {

class ImmediateConsoleLogger : public Logger {
public:
	ImmediateConsoleLogger();
	virtual ~ImmediateConsoleLogger();

	virtual Logger* clone();

	virtual void log(const char* str);
	virtual void log(const char* str, double d);
	virtual void flushBuffer();
};

}}}
#endif /* IMMEDIATECONSOLELOGGER_H_ */
