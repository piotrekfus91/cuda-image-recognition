#include <opencv2/core/types_c.h>
#include <vector_types.h>

namespace cir { namespace gpuprocessing {

void median_blur(uchar* origData, uchar* cloneData, int width, int height, int size, int step);

template <int SIZE>
__global__
void k_median_blur(uchar* origData, uchar* cloneData, int width, int height, int step);

}}
