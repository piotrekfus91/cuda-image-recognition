#include "gtest/gtest.h"
#include "cnn/file/CnnFileLoader.h"

using namespace cnn;
using namespace cnn::file;

class CnnFileLoaderTest : public ::testing::Test {

};

TEST_F(CnnFileLoaderTest, SimpleLoading) {
	CnnFileLoader fileLoader;
	NeuralNetwork* nn = fileLoader.load("file_saving_test.json");

	double* inputs = (double*) malloc(sizeof(double) * 2);
	inputs[0] = 3;
	inputs[1] = 5;

	nn->setInputs(inputs);
	nn->run();

	ASSERT_EQ(nn->getOutput(0), 84);
}
