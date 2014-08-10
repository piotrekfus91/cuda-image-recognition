#include "cnn/neuron/NeuronUtil.h"
#include "cnn/neuron/IdentityNeuron.h"
#include "cnn/neuron/SigmoidNeuron.h"
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
	if(neuronType == SIGMOID)
		return sizeof(SigmoidNeuron);
	throw InvalidNeuronTypeException();
}

Neuron* NeuronUtil::create(NeuronType neuronType, int inputNumber) {
	if(neuronType == IDENTITY)
		return new IdentityNeuron(inputNumber);
	if(neuronType == SIGMOID)
		return new SigmoidNeuron(inputNumber);
	throw InvalidNeuronTypeException();
}

}}
