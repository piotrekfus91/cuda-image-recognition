#ifndef DETECT_COLOR_CUH_
#define DETECT_COLOR_CUH_

#include <vector_types.h>
#include <opencv2/core/types_c.h>
#include "cir/common/Hsv.h"

namespace cir { namespace gpuprocessing {

void detect_color(uchar* src, const int hsvRangesNumber, const cir::common::OpenCvHsvRange* hsvRanges,
		const int width, const int height, const int step, uchar* dst, cudaStream_t stream);

__global__
void k_detect_color(uchar* src, const int hsvRangesNumber, const cir::common::OpenCvHsvRange* hsvRanges,
		const int width, const int height, const int step, uchar* dst);

}}

#endif
