#ifndef DETECT_COLOR_CUH_
#define DETECT_COLOR_CUH_

#include <vector_types.h>
#include <opencv2/core/types_c.h>

namespace cir { namespace gpuprocessing {

void detect_color(uchar* src, const int minHue, const int maxHue, const int minSat,
		const int maxSat, const int minValue, const int maxValue, const int width,
		const int height, const int step, uchar* dst);

__global__
void k_detect_color(uchar* src, const int minHue, const int maxHue, const int minSat,
		const int maxSat, const int minValue, const int maxValue, const int width,
		const int height, const int step, uchar* dst);

}}

#endif
