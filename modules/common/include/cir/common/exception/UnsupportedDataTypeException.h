#ifndef UNSUPPORTEDDATATYPEEXCEPTION_H_
#define UNSUPPORTEDDATATYPEEXCEPTION_H_

#include <exception>
#include <string>

namespace cir { namespace common { namespace exception {

class UnsupportedDataTypeException : std::exception {
public:
	const char* what();
private:
	static std::string MSG;
};

}}}
#endif /* UNSUPPORTEDDATATYPEEXCEPTION_H_ */
