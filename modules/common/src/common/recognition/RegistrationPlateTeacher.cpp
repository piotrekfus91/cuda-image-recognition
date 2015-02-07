#include "cir/common/recognition/RegistrationPlateTeacher.h"
#include "cir/common/exception/FileException.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/config.h"
#include "dirent.h"

using namespace std;
using namespace cir::common::exception;

namespace cir { namespace common { namespace recognition {

RegistrationPlateTeacher::RegistrationPlateTeacher(RegistrationPlateRecognizor* recognizor)
		: _recognizor(recognizor) {

}

RegistrationPlateTeacher::~RegistrationPlateTeacher() {

}

void RegistrationPlateTeacher::teach(string dirPath) {
	std::string extension = REGISTRATION_PLATE_PATTERN_EXTENSION;
	DIR* dir;
	if((dir = opendir(dirPath.c_str())) != NULL) {
		struct dirent *entry;
		while((entry = readdir(dir)) != NULL) {
			std::string fileName(entry->d_name);
			if(extension.length() >= fileName.length())
				continue;

			if(0 == fileName.compare(fileName.length() - extension.length(), extension.length(), extension)) {
				std::string filePath = dirPath;
				filePath.append(PATH_SEPARATOR);
				filePath.append(fileName);
				_recognizor->learn(filePath.c_str());
			}
		}
	} else {
		throw new FileException("Cannot open dir");
	}
}

}}}
