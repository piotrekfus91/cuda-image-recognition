#ifndef LAYER_H_
#define LAYER_H_

#include "cnn/neuron/Neuron.h"
#include "cnn/neuron/NeuronType.h"

namespace cnn {

class Layer {
public:
	Layer(const int neuronsNumber, cnn::neuron::NeuronType neuronType, int neuronInputNumber);
	virtual ~Layer();

	void setInputs(const int neuronIndex, double* inputs);
	void setWeights(const int neuronIndex, double* weights);
	const double getOutput(const int neuronIndex) const;

private:
	cnn::neuron::Neuron** _neurons;
	int _neuronsNumber;
	cnn::neuron::NeuronType _neuronType;
	int _neuronTypeSize;
};

}
#endif /* LAYER_H_ */
