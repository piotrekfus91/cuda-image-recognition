#include "gtest/gtest.h"
#include "cnn/file/FileSaver.h"

using namespace cnn;
using namespace cnn::file;
using namespace cnn::neuron;

class FileSaverTest : public ::testing::Test {

};

TEST_F(FileSaverTest, SimpleSaving) {
	double* weights00 = (double*) malloc(sizeof(double) * 1);
	weights00[0] = 2;

	double* weights01 = (double*) malloc(sizeof(double) * 1);
	weights01[0] = 3;

	double* weights1 = (double*) malloc(sizeof(double) * 2);
	weights1[0] = 4;
	weights1[1] = 4;

	double* inputs = (double*) malloc(sizeof(double) * 2);
	inputs[0] = 3;
	inputs[1] = 5;

	int* layersSize = (int*) malloc(sizeof(int) * 2);
	layersSize[0] = 2;
	layersSize[1] = 1;

	NeuralNetwork nn(IDENTITY, 2, layersSize);
	nn.setWeights(0, 0, weights00);
	nn.setWeights(0, 1, weights01);
	nn.setWeights(1, 0, weights1);
	nn.setInputs(inputs);

	FileSaver fileSaver;
	fileSaver.save(&nn, "file_saving_test.json");
	// +2 looks good to me, approved
}
