#include "cnn/neuron/NeuronUtil.h"
#include "cnn/neuron/IdentityNeuron.h"
#include "cnn/neuron/SigmoidNeuron.h"
#include "cnn/exception/InvalidNeuronTypeException.h"

using namespace cnn::exception;

namespace cnn { namespace neuron {

std::map<NeuronType, std::string> NeuronUtil::NEURON_NAMES;

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

std::string NeuronUtil::getName(NeuronType neuronType) {
	initNeuronNames();
	return NEURON_NAMES[neuronType];
}

NeuronType NeuronUtil::getType(std::string name) {
	initNeuronNames();
	std::map<NeuronType, std::string>::iterator iter = NEURON_NAMES.begin();
	while(iter != NEURON_NAMES.end()) {
		if(iter->second == name)
			return iter->first;
		++iter;
	}
	throw InvalidNeuronTypeException();
}

void NeuronUtil::initNeuronNames() {
	if(NEURON_NAMES.empty()) {
		NEURON_NAMES[IDENTITY] = "IDENTITY";
		NEURON_NAMES[SIGMOID] = "SIGMOID";
	}
}

}}
