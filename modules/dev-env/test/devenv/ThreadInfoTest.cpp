#include <gtest/gtest.h>
#include "cir/devenv/ThreadInfo.h"

using namespace cir::devenv;

class ThreadInfoTest : public ::testing::Test {

};

TEST_F(ThreadInfoTest, CanRunOpenCvGpuCode) {
	std::cout << "Number of threads: " << ThreadInfo::getNumberOfThreads() << std::endl;
	ASSERT_TRUE(ThreadInfo::getNumberOfThreads() > 0);
}
