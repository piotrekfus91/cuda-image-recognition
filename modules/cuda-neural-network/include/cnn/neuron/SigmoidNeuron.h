#ifndef SIGMOIDNEURON_H_
#define SIGMOIDNEURON_H_

#include "cnn/neuron/Neuron.h"

namespace cnn { namespace neuron {

class SigmoidNeuron : public Neuron {
public:
	SigmoidNeuron(int inputNumber);
	virtual ~SigmoidNeuron();

	virtual NeuronType getType();
	void setBeta(double beta);

protected:
	double _beta;
	static int DEFAULT_BETA;

	virtual void activate();
};

}}
#endif
