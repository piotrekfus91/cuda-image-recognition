#ifndef CONSOLECSVLOGGER_H_
#define CONSOLECSVLOGGER_H_

#include "cir/common/logger/CsvLogger.h"

namespace cir { namespace common { namespace logger {

class ConsoleCsvLogger : public CsvLogger {
public:
	ConsoleCsvLogger();
	virtual ~ConsoleCsvLogger();

	virtual Logger* clone();
	virtual void flushBuffer();
};

}}}
#endif /* CONSOLECSVLOGGER_H_ */
