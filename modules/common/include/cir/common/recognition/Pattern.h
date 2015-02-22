#ifndef PATTERN_H_
#define PATTERN_H_

#include <string>

namespace cir { namespace common { namespace recognition {

class Pattern {
public:
	Pattern();
	Pattern(std::string fileName, int segmentsNumber, double** huMoments);
	Pattern(const Pattern& pattern);
	virtual ~Pattern();

	std::string getFileName();
	void setFileName(std::string fileName);
	double getHuMoment(int segment, int index) const;
	double* getHuMoments(int segment) const;
	void setHuMoments(double** huMoments);
	void setSegmentsNumber(int segmentsNumber);

	const bool matches(int segment, double* huMoments) const;

private:
	std::string _fileName;
	int _segmentsNumber;
	double** _huMoments;
};

}}}
#endif /* PATTERN_H_ */
