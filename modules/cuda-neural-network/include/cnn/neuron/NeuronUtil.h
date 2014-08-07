#ifndef NEURONUTIL_H_
#define NEURONUTIL_H_

#include "cnn/neuron/NeuronType.h"
#include "cnn/neuron/Neuron.h"

namespace cnn { namespace neuron {

class NeuronUtil {
public:
	NeuronUtil();
	virtual ~NeuronUtil();

	static int sizeOf(NeuronType neuronType);
	static Neuron* create(NeuronType neuronType, int inputNumber);
};

}}
#endif /* NEURONUTIL_H_ */
