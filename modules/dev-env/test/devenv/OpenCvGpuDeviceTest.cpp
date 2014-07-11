#include <gtest/gtest.h>
#include "cir/devenv/OpenCvGpuChecker.h"

using namespace cir::devenv;

class OpenCvGpuDeviceTest : public ::testing::Test {
public:
	OpenCvGpuChecker openCvGpuChecker;
};

TEST_F(OpenCvGpuDeviceTest, CanRunOpenCvGpuCode) {
	ASSERT_TRUE(openCvGpuChecker.canRunOpenCvGpuCode());
}
