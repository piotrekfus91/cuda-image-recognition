#include <cstdlib>
#include <iostream>
#include "cnn/NeuralNetwork.h"

using namespace cnn::neuron;

namespace cnn {

NeuralNetwork::NeuralNetwork(const NeuronType neuronType, int layersNumber, int* layersSize)
		: _neuronType(neuronType), _layersNumber(layersNumber), _layersSize(layersSize), INPUT_LAYER(0),
		  OUTPUT_LAYER(_layersNumber-1) {
	_layers = (Layer**) malloc(sizeof(Layer*) * layersNumber);
	for(int i = 0; i < layersNumber; i++) {
		const int neuronInputNumber = getNeuronInputNumber(i);
		_layers[i] = new Layer(layersSize[i], _neuronType, neuronInputNumber);
	}
}

NeuralNetwork::~NeuralNetwork() {
	free(_layers);
}

const double NeuralNetwork::getOutput(const int neuronIndex) const {
	return _layers[OUTPUT_LAYER]->getOutput(neuronIndex);
}

void NeuralNetwork::setInputs(double* inputs) {
	Layer* inputLayer = _layers[INPUT_LAYER];
	for(int i = 0; i < inputLayer->getNeuronsNumber(); i++) {
		double* singleInput = (double*) malloc(sizeof(double) * 1);
		singleInput[0] = inputs[i];
		inputLayer->setInputs(i, singleInput);
	}
}

void NeuralNetwork::setWeights(const int layerIndex, const int neuronIndex, double* weights) {
	_layers[layerIndex]->setWeights(neuronIndex, weights);
}

void NeuralNetwork::run() {
	for(int i = 1; i < _layersNumber; i++) {
		// grab outputs from previous layer
		int inputsSize = _layersSize[i-1];
		double* inputs = (double*) malloc(sizeof(double) * inputsSize);
		for(int j = 0; j < inputsSize; j++) {
			inputs[j] = _layers[i-1]->getOutput(j);
		}

		// set as input to all neurons in current layer
		int layerSize = _layersSize[i];
		for(int j = 0; j < layerSize; j++) {
			_layers[i]->setInputs(j, inputs);
		}
	}
}

const int NeuralNetwork::getNeuronInputNumber(int layerIndex) const {
	if(layerIndex == 0)
		return 1;

	return _layersSize[layerIndex-1];
}

const int NeuralNetwork::getLayersNumber() const {
	return _layersNumber;
}

const NeuronType NeuralNetwork::getNeuronType() const {
	return _neuronType;
}

const int* NeuralNetwork::getLayersSize() const {
	return _layersSize;
}

const double* NeuralNetwork::getWeights(const int layerIndex, const int neuronIndex) const {
	return _layers[layerIndex]->getWeights(neuronIndex);
}

}
