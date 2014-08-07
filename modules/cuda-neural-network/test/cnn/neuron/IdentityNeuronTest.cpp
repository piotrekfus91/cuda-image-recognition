#include <gtest/gtest.h>
#include <cstdlib>
#include "cnn/neuron/IdentityNeuron.h"

using namespace cnn::neuron;

class IdentityNeuronTest : public ::testing::Test {
public:

};

TEST_F(IdentityNeuronTest, ActivationFunction) {
	const int INPUT_NUMBER = 2;
	IdentityNeuron neuron(INPUT_NUMBER);

	double* weights = (double*)malloc(sizeof(double) * INPUT_NUMBER);
	weights[0] = 1;
	weights[1] = 1;
	neuron.setWeights(weights);

	double* inputs = (double*)malloc(sizeof(double) * INPUT_NUMBER);
	inputs[0] = 2;
	inputs[1] = 3;
	neuron.setInputs(inputs);

	ASSERT_EQ(5, neuron.getOutput());

	free(inputs);
}
