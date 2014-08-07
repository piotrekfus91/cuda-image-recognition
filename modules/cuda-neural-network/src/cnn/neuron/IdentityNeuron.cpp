#include "cnn/neuron/IdentityNeuron.h"

namespace cnn { namespace neuron {

IdentityNeuron::IdentityNeuron(int inputNumber) : Neuron(inputNumber) {

}

IdentityNeuron::~IdentityNeuron() {

}

void IdentityNeuron::activate() {
	_output = sumAll();
}

NeuronType IdentityNeuron::getType() {
	return IDENTITY;
}

}}
