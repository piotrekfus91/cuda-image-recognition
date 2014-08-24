#include "gtest/gtest.h"
#include "cnn/file/CnnFileLoader.h"
#include <string>
#include "cir/common/config.h"

using namespace cnn;
using namespace cnn::file;

class CnnFileLoaderTest : public ::testing::Test {

};

TEST_F(CnnFileLoaderTest, SimpleLoading) {
	CnnFileLoader fileLoader;
	std::string path = TEST_FILES_DIR;
	path.append(PATH_SEPARATOR);
	path.append("cuda-neural-network");
	path.append(PATH_SEPARATOR);
	path.append("file_loading_test.json");

	NeuralNetwork* nn = fileLoader.load(path);

	double* inputs = (double*) malloc(sizeof(double) * 2);
	inputs[0] = 3;
	inputs[1] = 5;

	nn->setInputs(inputs);
	nn->run();

	ASSERT_EQ(nn->getOutput(0), 84);
}
