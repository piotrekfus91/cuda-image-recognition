#ifndef INVALIDMATTYPE_H_
#define INVALIDMATTYPE_H_

#include <exception>
#include <string>

namespace cir { namespace common { namespace exception {

class InvalidMatTypeException: public std::exception {
public:
	const char* what();
private:
	static std::string MSG;
};

}}}

#endif
