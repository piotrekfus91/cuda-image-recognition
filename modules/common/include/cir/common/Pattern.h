#ifndef PATTERN_H_
#define PATTERN_H_

#include <string>

namespace cir { namespace common {

class Pattern {
public:
	Pattern(std::string fileName, int segmentsNumber, double** huMoments);
	Pattern(const Pattern& pattern);
	virtual ~Pattern();

	std::string getFileName();
	double getHuMoment(int segment, int index);

private:
	std::string _fileName;
	int _segmentsNumber;
	double** _huMoments;
};

}}
#endif /* PATTERN_H_ */
