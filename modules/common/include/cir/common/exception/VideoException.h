#ifndef VIDEOEXCEPTION_H_
#define VIDEOEXCEPTION_H_

#include <string>
#include <exception>

namespace cir { namespace common { namespace exception {

class VideoException : std::exception {
public:
	VideoException(std::string msg);
	virtual ~VideoException() throw();

	const char* what();
private:
	std::string _msg;
};

}}}
#endif /* VIDEOEXCEPTION_H_ */
