#include <json/json.h>
#include <fstream>
#include "cnn/file/FileSaver.h"
#include "cnn/neuron/NeuronUtil.h"

using namespace cnn;
using namespace cnn::neuron;
using namespace Json;

namespace cnn { namespace file  {

FileSaver::FileSaver() {

}

FileSaver::~FileSaver() {

}

void FileSaver::save(NeuralNetwork* nn, std::string filePath) const {
	Value root;
	root["layers_number"] = nn->getLayersNumber();
	root["neuron_type"] = NeuronUtil::getName(nn->getNeuronType());

	Value layersSize(arrayValue);
	for(int i = 0; i < nn->getLayersNumber(); i++)
		layersSize.append(nn->getLayersSize()[i]);
	root["layers_size"] = layersSize;

	Value layersValue(arrayValue);

	for(int layerIndex = 0; layerIndex < nn->getLayersNumber(); layerIndex++) {
		Value layerValue;
		layerValue["layer_index"] = layerIndex;
		layerValue["neuron_number"] = nn->getLayersSize()[layerIndex];

		Value neuronsValue(arrayValue);
		for(int neuronIndex = 0; neuronIndex < nn->getLayersSize()[layerIndex]; neuronIndex++) {
			const double* weights = nn->getWeights(layerIndex, neuronIndex);
			Value weightsValue(arrayValue);
			for(int inputIndex = 0; inputIndex < nn->getNeuronInputNumber(layerIndex); inputIndex++) {
				weightsValue.append(weights[inputIndex]);
			}

			Value neuronValue;
			neuronValue["neuron_index"] = neuronIndex;
			neuronValue["input_number"] = nn->getNeuronInputNumber(layerIndex);
			neuronValue["weights"] = weightsValue;
			neuronsValue.append(neuronValue);
		}

		layerValue["neurons"] = neuronsValue;
		layersValue.append(layerValue);
	}

	root["layers"] = layersValue;

	std::ofstream outputFileStream;
	outputFileStream.open(filePath.c_str());
	outputFileStream << root;
	outputFileStream.close();
}

}}
