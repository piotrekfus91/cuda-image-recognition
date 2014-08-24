#include <fstream>
#include <cstdlib>
#include <json/json.h>
#include "cnn/file/CnnFileLoader.h"
#include "cnn/neuron/NeuronUtil.h"

using namespace cnn;
using namespace cnn::neuron;
using namespace Json;

namespace cnn { namespace file {

CnnFileLoader::CnnFileLoader() {

}

CnnFileLoader::~CnnFileLoader() {

}

NeuralNetwork* CnnFileLoader::load(std::string filePath) {
	Value root;

	std::ifstream inputFileStream;
	inputFileStream.open(filePath.c_str());
	inputFileStream >> root;
	inputFileStream.close();

	NeuronType neuronType = NeuronUtil::getType(root["neuron_type"].asCString());
	int layersNumber = root["layers_number"].asInt();

	Value layersSizeValue(arrayValue);
	layersSizeValue = root["layers_size"];
	int* layersSize = (int*) malloc(sizeof(int) * layersNumber);
	for(int i = 0; i < layersNumber; i++) {
		layersSize[i] = layersSizeValue.get(i, -1).asInt();
	}

	NeuralNetwork* nn = new NeuralNetwork(neuronType, layersNumber, layersSize);

	Value layersValue(arrayValue);
	layersValue = root["layers"];

	for(int i = 0; i < layersNumber; i++) {
		Value layerValue = layersValue[i];
		int neuronNumber = layerValue.get("neuron_number", -1).asInt();

		Value neuronsValue(arrayValue);
		neuronsValue = layerValue["neurons"];
		for(int j = 0; j < neuronNumber; j++) {
			Value neuronValue = neuronsValue[j];
			int inputNumber = neuronValue.get("input_number", -1).asInt();

			Value weightsValue(arrayValue);
			weightsValue = neuronValue["weights"];
			double* weights = (double*) malloc(sizeof(double) * inputNumber);
			for(unsigned int k = 0; k < weightsValue.size(); k++) {
				weights[k] = weightsValue.get(k, -1).asDouble();
			}

			nn->setWeights(i, j, weights);
		}
	}

	return nn;
}

}}
