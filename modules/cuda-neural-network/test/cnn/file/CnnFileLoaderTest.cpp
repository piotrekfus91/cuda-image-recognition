#include "gtest/gtest.h"
#include <string>
#include "cnn/file/CnnFileLoader.h"
#include "cir/common/test_file_loader.h"

using namespace cnn;
using namespace cnn::file;
using namespace cir::common;

class CnnFileLoaderTest : public ::testing::Test {

};

TEST_F(CnnFileLoaderTest, SimpleLoading) {
	CnnFileLoader fileLoader;
	std::string path = getTestFile("cuda-neural-network", "file_loading_test.json");

	NeuralNetwork* nn = fileLoader.load(path);

	double* inputs = (double*) malloc(sizeof(double) * 2);
	inputs[0] = 3;
	inputs[1] = 5;

	nn->setInputs(inputs);
	nn->run();

	ASSERT_EQ(nn->getOutput(0), 84);
}
