#include "cir/common/Hsv.h"

namespace cir { namespace common {

HsvRange getWhiteRange() {
	Hsv lessWHite;
	lessWHite.hue = 0;
	lessWHite.saturation = 0.01;
	lessWHite.value = 0.6;
	Hsv greaterWhite;
	greaterWhite.hue = 360;
	greaterWhite.saturation = 0.3;
	greaterWhite.value = 1.0;
	HsvRange rangeWhite;
	rangeWhite.less = lessWHite;
	rangeWhite.greater = greaterWhite;
	return rangeWhite;
}

HsvRange getRedRange() {
	Hsv lessRed;
	lessRed.hue = 330;
	lessRed.saturation = 0.2;
	lessRed.value = 0.2;
	Hsv greaterRed;
	greaterRed.hue = 30;
	greaterRed.saturation = 1;
	greaterRed.value = 1;
	HsvRange rangeRed;
	rangeRed.less = lessRed;
	rangeRed.greater = greaterRed;
	return rangeRed;
}

HsvRange getBlueRange() {
	Hsv lessBlue;
	lessBlue.hue = 200;
	lessBlue.saturation = 0.4;
	lessBlue.value = 0.2;
	Hsv greaterBlue;
	greaterBlue.hue = 255;
	greaterBlue.saturation = 1;
	greaterBlue.value = 1;
	HsvRange rangeBlue;
	rangeBlue.less = lessBlue;
	rangeBlue.greater = greaterBlue;
	return rangeBlue;
}

HsvRange getYellowRange() {
	Hsv lessYellow;
	lessYellow.hue = 45;
	lessYellow.saturation = 0.4;
	lessYellow.value = 0.2;
	Hsv greaterYellow;
	greaterYellow.hue = 75;
	greaterYellow.saturation = 1;
	greaterYellow.value = 1;
	HsvRange rangeYellow;
	rangeYellow.less = lessYellow;
	rangeYellow.greater = greaterYellow;
	return rangeYellow;
}

}}
