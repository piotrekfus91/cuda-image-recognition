#include <iostream>
#include "cir/common/cuda_host_util.cuh"

namespace cir { namespace common {

cir::common::logger::Logger* default_logger;

void cuda_init() {

}

void cuda_shutdown() {

}

void cuda_handle_error(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
    	std::cerr << "ERROR!" << std::endl;
    	std::cerr << file << ":" << line << std::endl;
    	std::cerr << err << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void set_default_logger(cir::common::logger::Logger* logger) {
	default_logger = logger;
}

cir::common::logger::Logger* get_default_logger() {
	return default_logger;
}

}}
