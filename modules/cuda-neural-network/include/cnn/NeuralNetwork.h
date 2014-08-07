#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "cnn/neuron/NeuronType.h"
#include "cnn/Layer.h"

namespace cnn {

class NeuralNetwork {
public:
	NeuralNetwork(const cnn::neuron::NeuronType neuronType, int layersNumber, int* layersSize);
	virtual ~NeuralNetwork();

	void setInputs(double* inputs);
	void setWeights(const int layerIndex, const int neuronIndex, double* weights);
	const double getOutput(const int neuronIndex);

	void run();

private:
	cnn::neuron::NeuronType _neuronType;
	cnn::Layer** _layers;
	int _layersNumber;
	int* _layersSize;
	int INPUT_LAYER;
	int OUTPUT_LAYER;

	const int getNeuronInputNumber(int layerIndex) const;
};

}
#endif /* NEURALNETWORK_H_ */
