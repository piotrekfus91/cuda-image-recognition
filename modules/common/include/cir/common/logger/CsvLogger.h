#ifndef CSVLOGGER_H_
#define CSVLOGGER_H_

#include <sstream>
#include "cir/common/logger/Logger.h"

namespace cir { namespace common { namespace logger {

class CsvLogger : public Logger {
public:
	CsvLogger();
	virtual ~CsvLogger();

	virtual Logger* clone() = 0;

	virtual void log(const char* str);
	virtual void log(const char* str, double d);
	virtual void flushBuffer() = 0;

protected:
	std::stringstream _buffer;
	static char SEPARATOR;
};

}}}
#endif /* CSVLOGGER_H_ */
