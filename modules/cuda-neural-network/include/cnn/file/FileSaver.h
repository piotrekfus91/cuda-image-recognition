#ifndef FILESAVER_H_
#define FILESAVER_H_

#include <cstring>
#include "cnn/NeuralNetwork.h"

namespace cnn { namespace file {

class FileSaver {
public:
	FileSaver();
	virtual ~FileSaver();

	void save(cnn::NeuralNetwork* nn, std::string filePath) const;
};

}
}
#endif
