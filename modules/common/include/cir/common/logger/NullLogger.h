#ifndef NULLLOGGER_H_
#define NULLLOGGER_H_

#include "cir/common/logger/Logger.h"

namespace cir { namespace common { namespace logger {

class NullLogger : public Logger {
public:
	NullLogger();
	virtual ~NullLogger();

	virtual Logger* clone();

	virtual void log(const char* str);
	virtual void log(const char* str, double d);
	virtual void flushBuffer();
};

}}}
#endif /* NULLLOGGER_H_ */
