#include "cnn/exception/InvalidNeuronTypeException.h"

using namespace std;

namespace cnn { namespace exception {

const char* InvalidNeuronTypeException::what() {
	return MSG.c_str();
}

static string MSG = "Invalid neuron type!";

}}
