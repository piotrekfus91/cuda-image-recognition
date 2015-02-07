#ifndef REGISTRATIONPLATETEACHER_H_
#define REGISTRATIONPLATETEACHER_H_

#include <string>
#include "cir/common/recognition/RegistrationPlateRecognizor.h"

namespace cir { namespace common { namespace recognition {

class RegistrationPlateTeacher {
public:
	RegistrationPlateTeacher(RegistrationPlateRecognizor* recognizor);
	virtual ~RegistrationPlateTeacher();

	virtual void teach(std::string dir);

private:
	RegistrationPlateRecognizor* _recognizor;
};

}}}
#endif /* REGISTRATIONPLATETEACHER_H_ */
