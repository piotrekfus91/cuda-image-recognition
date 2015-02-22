#ifndef HSV_H_
#define HSV_H_

namespace cir { namespace common {

struct Hsv {
	float hue;
	float saturation;
	float value;
};

struct HsvRange {
	Hsv less;
	Hsv greater;
};

struct OpenCvHsv {
	int hue;
	int saturation;
	int value;
};

struct OpenCvHsvRange {
	OpenCvHsv less;
	OpenCvHsv greater;
};

HsvRange getWhiteRange();
HsvRange getRedRange();
HsvRange getBlueRange();
HsvRange getYellowRange();

}}

#endif /* HSV_H_ */
