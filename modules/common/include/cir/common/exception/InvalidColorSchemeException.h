#ifndef INVALIDCOLORSCHEMEEXCEPTION_H_
#define INVALIDCOLORSCHEMEEXCEPTION_H_

#include <exception>
#include <string>

namespace cir { namespace common { namespace exception {

class InvalidColorSchemeException: public std::exception {
public:
	const char* what();
private:
	static std::string MSG;
};

}}}
#endif /* INVALIDCOLORSCHEMEEXCEPTION_H_ */
