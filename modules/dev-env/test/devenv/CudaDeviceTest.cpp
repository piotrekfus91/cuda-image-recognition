#include <gtest/gtest.h>
#include "cir/devenv/CudaDeviceChecker.h"

using namespace cir::devenv;

class CudaDeviceTest : public ::testing::Test {
public:
	CudaDeviceChecker cudaDeviceChecker;
};

TEST_F(CudaDeviceTest, CanRunNativeCudaCode) {
	ASSERT_TRUE(cudaDeviceChecker.canRunNativeCudaCode());
}
