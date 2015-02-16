#ifndef OCREXCEPTION_H_
#define OCREXCEPTION_H_

#include <exception>
#include <string>

namespace cir { namespace common { namespace exception {

class OcrException : std::exception {
public:
	const char* what();
private:
	static std::string MSG;
};

}}}
#endif /* OCREXCEPTION_H_ */
