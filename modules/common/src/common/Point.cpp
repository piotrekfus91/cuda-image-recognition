#include "cir/common/Point.h"

namespace cir { namespace common {

Point createPoint(int x, int y) {
	Point point;
	point.x = x;
	point.y = y;
	return point;
}

}}
