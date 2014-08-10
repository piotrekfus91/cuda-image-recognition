#ifndef NEURON_H_
#define NEURON_H_

#include "cnn/neuron/NeuronType.h"

namespace cnn { namespace neuron {

class Neuron {
public:
	Neuron(int inputNumber);
	virtual ~Neuron();

	const double getOutput() const;
	void setInputs(double* inputs);
	void setWeights(double* weights);
	const double* getWeights() const;

	virtual NeuronType getType() = 0;

protected:
	int _inputNumber;
	double _output;
	double* _weights;
	double* _inputs;
	bool _weightsInitializedInternally;
	double sumAll() const;
	void generateRandomWeights();

	virtual void activate() = 0;
};

}}
#endif /* NEURON_H_ */
