#ifndef COUNT_MOMENTS_CUH_
#define COUNT_MOMENTS_CUH_

#include <vector_types.h>
#include <opencv2/core/types_c.h>

namespace cir { namespace gpuprocessing {

void count_raw_moment_init(int width, int height);

// http://en.wikipedia.org/wiki/Image_moment
double count_raw_moment(uchar* data, int width, int height, int step, int p, int q);

void count_raw_moment_shutdown();

__global__
void k_count_raw_moment(uchar* data, int width, int height, int step, int p, int q, double* result);

__device__
int pow(int p, int q);

}}

#endif /* COUNT_MOMENTS_CUH_ */
