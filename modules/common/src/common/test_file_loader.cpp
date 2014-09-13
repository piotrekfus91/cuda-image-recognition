#include "cir/common/test_file_loader.h"
#include "cir/common/config.h"

namespace cir { namespace common {

std::string getTestFile(std::string module, std::string fileName) {
	std::string path = TEST_FILES_DIR;
	path.append(PATH_SEPARATOR);
	path.append(module);
	path.append(PATH_SEPARATOR);
	path.append(fileName);
	return path;
}

}}
