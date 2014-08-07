#include <gtest/gtest.h>
#include <cstdlib>
#include "cnn/Layer.h"
#include "cnn/neuron/NeuronType.h"

using namespace cnn;

class LayerTest : public ::testing::Test {
public:

};

TEST_F(LayerTest, TwoNeurons) {
	const int INPUT_NUMBER = 2;
	const int NEURON_NUMBER = 2;
	Layer layer(NEURON_NUMBER, cnn::neuron::IDENTITY, INPUT_NUMBER);

	double* weights0 = (double*)malloc(sizeof(double) * INPUT_NUMBER);
	weights0[0] = 0.5;
	weights0[1] = 1;
	layer.setWeights(0, weights0);

	double* weights1 = (double*)malloc(sizeof(double) * INPUT_NUMBER);
	weights1[0] = 1;
	weights1[1] = 0.5;
	layer.setWeights(1, weights1);

	double* inputs = (double*)malloc(sizeof(double) * INPUT_NUMBER);
	inputs[0] = 2;
	inputs[1] = 3;
	layer.setInputs(0, inputs);
	layer.setInputs(1, inputs);

	double output0 = layer.getOutput(0);
	double output1 = layer.getOutput(1);

	free(inputs);
	free(weights0);
	free(weights1);

	ASSERT_EQ(4, output0);
	ASSERT_EQ(3.5, output1);
}
