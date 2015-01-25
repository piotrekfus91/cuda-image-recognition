#ifndef INVALIDPARAMEXCEPTION_H_
#define INVALIDPARAMEXCEPTION_H_

#include <string>
#include <exception>

namespace cir { namespace common { namespace exception {

class InvalidParamException : public std::exception {
public:
	InvalidParamException(std::string msg);
	virtual ~InvalidParamException() throw();

	const char* what();
private:
	std::string _msg;
};

}}}
#endif /* INVALIDPARAMEXCEPTION_H_ */
