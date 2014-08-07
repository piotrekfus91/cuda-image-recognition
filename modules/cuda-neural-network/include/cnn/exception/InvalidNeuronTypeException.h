#ifndef INVALIDNEURONTYPEEXCEPTION_H_
#define INVALIDNEURONTYPEEXCEPTION_H_

#include <exception>
#include <string>

namespace cnn { namespace exception {

class InvalidNeuronTypeException : public std::exception {
public:
	const char* what();

private:
	static std::string MSG;
};

}}
#endif /* INVALIDNEURONTYPEEXCEPTION_H_ */
