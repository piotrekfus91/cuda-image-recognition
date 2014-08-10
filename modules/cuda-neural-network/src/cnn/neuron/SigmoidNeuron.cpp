#include <cmath>
#include "cnn/neuron/SigmoidNeuron.h"

namespace cnn { namespace neuron {

int SigmoidNeuron::DEFAULT_BETA = 0.8;

SigmoidNeuron::SigmoidNeuron(int inputNumber) : Neuron(inputNumber), _beta(DEFAULT_BETA) {

}

SigmoidNeuron::~SigmoidNeuron() {

}

NeuronType SigmoidNeuron::getType() {
	return SIGMOID;
}

void SigmoidNeuron::activate() {
	double sum = sumAll();
	_output = 1.0 / (1.0 + exp(-sum));
}

void SigmoidNeuron::setBeta(double beta) {
	_beta = beta;
}

}}
