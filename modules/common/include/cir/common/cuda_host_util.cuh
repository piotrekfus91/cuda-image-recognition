#ifndef CUDA_INIT_CUH_
#define CUDA_INIT_CUH_

#include <driver_types.h>

namespace cir { namespace common {

void cuda_init();
void cuda_shutdown();
void cuda_handle_error(cudaError_t err, const char *file, int line);

}}

#define HANDLE_CUDA_ERROR(err) (cir::common::cuda_handle_error(err, __FILE__, __LINE__))

#endif
