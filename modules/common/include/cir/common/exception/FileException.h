#ifndef FILEEXCEPTION_H_
#define FILEEXCEPTION_H_

#include <exception>

namespace cir { namespace common { namespace exception {

class FileException : std::exception {
public:
	FileException(const char* what);
	virtual ~FileException() throw();

	const char* what();
private:
	const char* _what;
};

}}}
#endif /* FILEEXCEPTION_H_ */
