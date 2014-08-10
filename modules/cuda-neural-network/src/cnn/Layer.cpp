#include <cstdlib>
#include <iostream>
#include "cnn/Layer.h"
#include "cnn/neuron/NeuronUtil.h"

using namespace cnn::neuron;

namespace cnn {

Layer::Layer(const int neuronsNumber, NeuronType neuronType, int neuronInputNumber)
		: _neuronsNumber(neuronsNumber), _neuronType(neuronType), _neuronInputNumber(neuronInputNumber) {
	_neuronTypeSize = NeuronUtil::sizeOf(neuronType);
	_neurons = (Neuron**) malloc(sizeof(Neuron*) * _neuronsNumber);
	for(int i = 0; i < _neuronsNumber; i++) {
		_neurons[i] = NeuronUtil::create(neuronType, neuronInputNumber);
	}
}

Layer::~Layer() {
	for(int i = 0; i < _neuronsNumber; i++) {
		free(_neurons[i]);
	}
	free(_neurons);
}

void Layer::setInputs(const int neuronIndex, double* inputs) {
	_neurons[neuronIndex]->setInputs(inputs);
}

void Layer::setWeights(const int neuronIndex, double* weights) {
	_neurons[neuronIndex]->setWeights(weights);
}

const double Layer::getOutput(const int neuronIndex) const {
	return _neurons[neuronIndex]->getOutput();
}

const int Layer::getNeuronsNumber() const {
	return _neuronsNumber;
}

const double* Layer::getWeights(const int neuronIndex) const {
	return _neurons[neuronIndex]->getWeights();
}

const int Layer::getNeuronInputNumber() const {
	return _neuronInputNumber;
}

}
