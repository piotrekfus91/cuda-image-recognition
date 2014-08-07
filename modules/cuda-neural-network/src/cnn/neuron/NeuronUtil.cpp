#include "cnn/neuron/NeuronUtil.h"
#include "cnn/neuron/IdentityNeuron.h"
#include "cnn/exception/InvalidNeuronTypeException.h"

using namespace cnn::exception;

namespace cnn { namespace neuron {

NeuronUtil::NeuronUtil() {

}

NeuronUtil::~NeuronUtil() {

}

int NeuronUtil::sizeOf(NeuronType neuronType) {
	if(neuronType == IDENTITY)
		return sizeof(IdentityNeuron);
	throw InvalidNeuronTypeException();
}

Neuron* NeuronUtil::create(NeuronType neuronType, int inputNumber) {
	if(neuronType == IDENTITY)
		return new IdentityNeuron(inputNumber);
	throw InvalidNeuronTypeException();
}

}}
