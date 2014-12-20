#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include "cir/common/test_file_loader.h"
#include "cir/common/logger/NullLogger.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/cpuprocessing/CpuImageProcessingService.h"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::cpuprocessing;
using namespace cir::gpuprocessing;

class GpuRegionSplittingSegmentatorTest : public ::testing::Test {
protected:
	GpuImageProcessingService* _service;
	Logger* _logger;

	virtual void SetUp();
	virtual void TearDown();
};

void GpuRegionSplittingSegmentatorTest::SetUp() {
	_logger = new NullLogger;
	_service = new GpuImageProcessingService(*_logger);
	_service->setSegmentatorMinSize(0);
}

void GpuRegionSplittingSegmentatorTest::TearDown() {
	delete _logger;
	delete _service;
}

TEST_F(GpuRegionSplittingSegmentatorTest, Sample) {
	cv::Mat mat = cv::imread(getTestFile("cpu-processing", "sample.bmp"));
	cv::gpu::GpuMat gpuMat(mat);
	MatWrapper mw(gpuMat);

	_service->init(mat.cols, mat.rows);

	mw = _service->bgrToHsv(mw);

	SegmentArray* segmentArray = _service->segmentate(mw);
	ASSERT_EQ(segmentArray->size, 8);

	Segment* segment = segmentArray->segments[0];
	ASSERT_EQ(segment->leftX, 0);
	ASSERT_EQ(segment->rightX, 0);
	ASSERT_EQ(segment->topY, 0);
	ASSERT_EQ(segment->bottomY, 0);

	segment = segmentArray->segments[1];
	ASSERT_EQ(segment->leftX, 2);
	ASSERT_EQ(segment->rightX, 6);
	ASSERT_EQ(segment->topY, 0);
	ASSERT_EQ(segment->bottomY, 1);

	segment = segmentArray->segments[2];
	ASSERT_EQ(segment->leftX, 7);
	ASSERT_EQ(segment->rightX, 7);
	ASSERT_EQ(segment->topY, 1);
	ASSERT_EQ(segment->bottomY, 1);

	segment = segmentArray->segments[3];
	ASSERT_EQ(segment->leftX, 0);
	ASSERT_EQ(segment->rightX, 4);
	ASSERT_EQ(segment->topY, 2);
	ASSERT_EQ(segment->bottomY, 7);

	segment = segmentArray->segments[4];
	ASSERT_EQ(segment->leftX, 5);
	ASSERT_EQ(segment->rightX, 6);
	ASSERT_EQ(segment->topY, 3);
	ASSERT_EQ(segment->bottomY, 3);

	segment = segmentArray->segments[5];
	ASSERT_EQ(segment->leftX, 7);
	ASSERT_EQ(segment->rightX, 7);
	ASSERT_EQ(segment->topY, 4);
	ASSERT_EQ(segment->bottomY, 4);

	segment = segmentArray->segments[6];
	ASSERT_EQ(segment->leftX, 7);
	ASSERT_EQ(segment->rightX, 7);
	ASSERT_EQ(segment->topY, 6);
	ASSERT_EQ(segment->bottomY, 6);

	segment = segmentArray->segments[7];
	ASSERT_EQ(segment->leftX, 5);
	ASSERT_EQ(segment->rightX, 5);
	ASSERT_EQ(segment->topY, 7);
	ASSERT_EQ(segment->bottomY, 7);
}

TEST_F(GpuRegionSplittingSegmentatorTest, Sample2) {
	cv::Mat mat = cv::imread(getTestFile("cpu-processing", "sample2.bmp"));
	cv::gpu::GpuMat gpuMat(mat);
	MatWrapper mw(gpuMat);

	_service->init(mat.cols, mat.rows);

	mw = _service->bgrToHsv(mw);
	SegmentArray* segmentArray = _service->segmentate(mw);
	ASSERT_EQ(segmentArray->size, 1);

	Segment* segment = segmentArray->segments[0];
	ASSERT_EQ(segment->leftX, 0);
	ASSERT_EQ(segment->rightX, 7);
	ASSERT_EQ(segment->topY, 0);
	ASSERT_EQ(segment->bottomY, 7);
}

TEST_F(GpuRegionSplittingSegmentatorTest, Sample9x11) {
	cv::Mat mat = cv::imread(getTestFile("cpu-processing", "sample9x11.bmp"));
	cv::gpu::GpuMat gpuMat(mat);
	MatWrapper mw(gpuMat);

	_service->init(mat.cols, mat.rows);

	mw = _service->bgrToHsv(mw);

	SegmentArray* segmentArray = _service->segmentate(mw);
	ASSERT_EQ(segmentArray->size, 9);

	Segment* segment = segmentArray->segments[0];
	ASSERT_EQ(segment->leftX, 0);
	ASSERT_EQ(segment->rightX, 0);
	ASSERT_EQ(segment->topY, 0);
	ASSERT_EQ(segment->bottomY, 0);

	segment = segmentArray->segments[1];
	ASSERT_EQ(segment->leftX, 2);
	ASSERT_EQ(segment->rightX, 6);
	ASSERT_EQ(segment->topY, 0);
	ASSERT_EQ(segment->bottomY, 1);

	segment = segmentArray->segments[2];
	ASSERT_EQ(segment->leftX, 7);
	ASSERT_EQ(segment->rightX, 7);
	ASSERT_EQ(segment->topY, 1);
	ASSERT_EQ(segment->bottomY, 1);

	segment = segmentArray->segments[3];
	ASSERT_EQ(segment->leftX, 0);
	ASSERT_EQ(segment->rightX, 3);
	ASSERT_EQ(segment->topY, 2);
	ASSERT_EQ(segment->bottomY, 8);

	segment = segmentArray->segments[4];
	ASSERT_EQ(segment->leftX, 5);
	ASSERT_EQ(segment->rightX, 6);
	ASSERT_EQ(segment->topY, 3);
	ASSERT_EQ(segment->bottomY, 3);

	segment = segmentArray->segments[5];
	ASSERT_EQ(segment->leftX, 7);
	ASSERT_EQ(segment->rightX, 8);
	ASSERT_EQ(segment->topY, 4);
	ASSERT_EQ(segment->bottomY, 4);

	segment = segmentArray->segments[6];
	ASSERT_EQ(segment->leftX, 7);
	ASSERT_EQ(segment->rightX, 8);
	ASSERT_EQ(segment->topY, 6);
	ASSERT_EQ(segment->bottomY, 6);

	segment = segmentArray->segments[7];
	ASSERT_EQ(segment->leftX, 5);
	ASSERT_EQ(segment->rightX, 6);
	ASSERT_EQ(segment->topY, 7);
	ASSERT_EQ(segment->bottomY, 10);

	segment = segmentArray->segments[8];
	ASSERT_EQ(segment->leftX, 0);
	ASSERT_EQ(segment->rightX, 0);
	ASSERT_EQ(segment->topY, 8);
	ASSERT_EQ(segment->bottomY, 8);
}

void compare_using_region_growing_and_splitting(const char* fileName, GpuImageProcessingService* _service, Logger* _logger) {
	cv::Mat mat = cv::imread(getTestFile("cpu-processing", fileName));
	cv::gpu::GpuMat gpuMat(mat);
	MatWrapper mw(gpuMat);

	_service->init(mat.cols, mat.rows);

	mw = _service->bgrToHsv(mw);
	SegmentArray* segmentArray = _service->segmentate(mw);

	CpuImageProcessingService regionGrowingService(*_logger);
	regionGrowingService.setSegmentator(new CpuRegionGrowingSegmentator);
	regionGrowingService.setSegmentatorMinSize(0);

	SegmentArray* correctSegmentArray = regionGrowingService.segmentate(MatWrapper(mat));

	for(int i = 0; i < segmentArray->size; i++) {
		Segment* segmentToTest = segmentArray->segments[i];
		bool found = false;
		for(int j = 0; j < correctSegmentArray->size; j++) {
			Segment* correctSegment = correctSegmentArray->segments[j];
			if(segmentToTest->leftX == correctSegment->leftX
					&& segmentToTest->rightX == correctSegment->rightX
					&& ((segmentToTest->topY == correctSegment->topY && segmentToTest->bottomY == correctSegment->bottomY)
							|| (segmentToTest->bottomY == correctSegment->topY && segmentToTest->topY == correctSegment->bottomY))) {
				found = true;
				break;
			}
		}
		if(!found) {
			std::cerr << i << ": " << "leftX=" << segmentToTest->leftX << ", topY=" << segmentToTest->topY <<
					", rightX=" << segmentToTest->rightX << ", bottomY=" << segmentToTest->bottomY << std::endl;
			ASSERT_TRUE(false);
		}
	}
}

TEST_F(GpuRegionSplittingSegmentatorTest, Metro) {
	compare_using_region_growing_and_splitting("metro_red_yellow.bmp", _service, _logger);
}
