#ifndef FILELOADER_H_
#define FILELOADER_H_

#include <string>
#include "cnn/NeuralNetwork.h"

namespace cnn { namespace file {

class FileLoader {
public:
	FileLoader();
	virtual ~FileLoader();

	virtual cnn::NeuralNetwork load(std::string filePath) = 0;
};

}}
#endif
