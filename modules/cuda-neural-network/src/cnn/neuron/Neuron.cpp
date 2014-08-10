#include <cstdlib>
#include <iostream>
#include "cnn/neuron/Neuron.h"

namespace cnn { namespace neuron {

Neuron::Neuron(int inputNumber) : _inputNumber(inputNumber), _output(0) {
	_weights = (double*)malloc(sizeof(double) * _inputNumber);
	_weightsInitializedInternally = true;
	generateRandomWeights();
}

Neuron::~Neuron() {
	if(_weightsInitializedInternally)
		free(_weights);
}

const double Neuron::getOutput() const {
	return _output;
}

void Neuron::setWeights(double* weights) {
	_weightsInitializedInternally = false;
	_weights = weights;
}

const double* Neuron::getWeights() const {
	return _weights;
}

void Neuron::setInputs(double* inputs) {
	_inputs = inputs;
	activate();
}

double Neuron::sumAll() const {
	double result = 0;
	for(int i = 0; i < _inputNumber; i++) {
		result += _inputs[i] * _weights[i];
	}
	return result;
}

void Neuron::generateRandomWeights() {
	// TODO random weights
	_weightsInitializedInternally = true;
	for(int i = 0; i < _inputNumber; i++) {
		_weights[i] = 0.5;
	}
}

}}
