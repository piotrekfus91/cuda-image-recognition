#ifndef COUNT_MOMENTS_CUH_
#define COUNT_MOMENTS_CUH_

#include <vector_types.h>
#include <opencv2/core/types_c.h>

namespace cir { namespace gpuprocessing {

// http://en.wikipedia.org/wiki/Image_moment
void count_raw_moments(uchar* data, int width, int height, int step, double* rawMoments,
		cudaStream_t stream);

__global__
void k_count_raw_moment(uchar* data, int width, int height, int step, int pass, long* result);

}}

#endif /* COUNT_MOMENTS_CUH_ */
