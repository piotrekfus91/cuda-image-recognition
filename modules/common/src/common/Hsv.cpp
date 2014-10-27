#include "cir/common/Hsv.h"

namespace cir { namespace common {

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

}}
