#ifndef LOGGER_H_
#define LOGGER_H_

#include <string>

namespace cir { namespace common { namespace logger {

class Logger {
public:
	Logger();
	virtual ~Logger();

	virtual Logger* clone() = 0;

	virtual void log(const char* str) = 0;
	virtual void log(const char* str, double d) = 0;
	virtual void flushBuffer() = 0;

	virtual void setModule(const char* module);

protected:
	std::string _module;
};

}}}
#endif /* TIMELOGGER_H_ */
