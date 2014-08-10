#ifndef CNNFILELOADER_H_
#define CNNFILELOADER_H_

#include <string>
#include "cnn/NeuralNetwork.h"

namespace cnn { namespace file {

class CnnFileLoader {
public:
	CnnFileLoader();
	virtual ~CnnFileLoader();

	virtual cnn::NeuralNetwork* load(std::string filePath);
};

}}
#endif
