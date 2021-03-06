#include "cir/common/config.h"
#include "cir/gpuprocessing/segmentate_base.cuh"

using namespace cir::common;

namespace cir { namespace gpuprocessing {

int _min_size = SEGMENTATOR_MIN_SIZE;

void set_segment_min_size(int minSize) {
	_min_size = minSize;
}

}}
