#ifndef CUDA_INIT_CUH_
#define CUDA_INIT_CUH_

#include <driver_types.h>
#include "cir/common/logger/Logger.h"

namespace cir { namespace common {

void cuda_init();
void cuda_shutdown();
void cuda_handle_error(cudaError_t err, const char *file, int line);
void set_default_logger(cir::common::logger::Logger* logger);
cir::common::logger::Logger* get_default_logger();

}}

#define HANDLE_CUDA_ERROR(err) (cir::common::cuda_handle_error(err, __FILE__, __LINE__))


#define KERNEL_MEASURE_START(stream) cudaEvent_t start; \
		cudaEvent_t stop; \
		HANDLE_CUDA_ERROR(cudaEventCreate(&start)); \
		HANDLE_CUDA_ERROR(cudaEventCreate(&stop)); \
		HANDLE_CUDA_ERROR(cudaEventRecord(start, stream));

#define KERNEL_MEASURE_END(msg, stream) HANDLE_CUDA_ERROR(cudaEventRecord(stop, stream)); \
		HANDLE_CUDA_ERROR(cudaEventRecord(stop, stream)); \
		HANDLE_CUDA_ERROR(cudaEventSynchronize(stop)); \
		\
		float time; \
		HANDLE_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop)); \
		HANDLE_CUDA_ERROR(cudaEventDestroy(start)); \
		HANDLE_CUDA_ERROR(cudaEventDestroy(stop)); \
		\
		time = time / 1000; \
		\
		Logger* logger = get_default_logger(); \
		logger->log(msg, time);

#endif
