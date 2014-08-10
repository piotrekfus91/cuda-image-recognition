#ifndef NEURONUTIL_H_
#define NEURONUTIL_H_

#include <map>
#include <string>
#include "cnn/neuron/NeuronType.h"
#include "cnn/neuron/Neuron.h"

namespace cnn { namespace neuron {

class NeuronUtil {
public:
	NeuronUtil();
	virtual ~NeuronUtil();

	static int sizeOf(NeuronType neuronType);
	static Neuron* create(NeuronType neuronType, int inputNumber);
	static std::string getName(NeuronType neuronType);
	static NeuronType getType(std::string name);

	static std::map<NeuronType, std::string> NEURON_NAMES;
	static void initNeuronNames();
};

}}
#endif /* NEURONUTIL_H_ */
