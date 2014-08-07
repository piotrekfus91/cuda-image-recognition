#ifndef IDENTITYNEURON_H_
#define IDENTITYNEURON_H_

#include "cnn/neuron/Neuron.h"

namespace cnn { namespace neuron {

class IdentityNeuron : public Neuron {
public:
	IdentityNeuron(int inputNumber);
	virtual ~IdentityNeuron();

	virtual NeuronType getType();

protected:
	virtual void activate();
};

}}
#endif /* IDENTITYNEURON_H_ */
