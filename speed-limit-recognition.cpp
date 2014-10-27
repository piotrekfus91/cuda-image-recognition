#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/recognition/SpeedLimitRecognizor.h"
#include "cir/common/test_file_loader.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::cpuprocessing;

int main(int argc, char** argv) {
	cv::namedWindow("Speed limit");

	ImmediateConsoleLogger logger;
	CpuImageProcessingService service(logger);

	cv::Mat mat = cv::imread(getTestFile("speed-limit", "table.jpg"));
	cv::imshow("Speed limit", mat);
	cv::waitKey(0);
	MatWrapper mw(mat);

	SpeedLimitRecognizor recognizor(service);
	recognizor.learn(cir::common::getTestFile("speed-limit", "pattern.jpg").c_str());
	recognizor.learn(cir::common::getTestFile("speed-limit", "pattern_small.jpg").c_str());
	RecognitionInfo recognitionInfo = recognizor.recognize(mw);
	mw = service.mark(mw, recognitionInfo.getMatchedSegments());
	cv::imshow("Speed limit", mw.getMat());

	cv::Mat bobrowiceMat = cv::imread(getTestFile("speed-limit", "bobrowice.jpg"));
	cv::Mat uszkodzonaJezdniaMat = cv::imread(getTestFile("speed-limit", "uszkodzona_jezdnia.jpg"));
	cv::Mat mat30 = cv::imread(getTestFile("speed-limit", "30.jpg"));
	cv::Mat mat40 = cv::imread(getTestFile("speed-limit", "40.jpg"));
	cv::Mat wiesMat = cv::imread(getTestFile("speed-limit", "wies.jpg"));
	cv::Mat zimaMat = cv::imread(getTestFile("speed-limit", "zima.jpg"));

	MatWrapper bobrowiceMw(bobrowiceMat);
	MatWrapper uszkodzonaJezdniaMw(uszkodzonaJezdniaMat);
	MatWrapper mw30(mat30);
	MatWrapper mw40(mat40);
	MatWrapper wiesMw(wiesMat);
	MatWrapper zimaMw(zimaMat);

	RecognitionInfo bobrowiceRecognitionInfo = recognizor.recognize(bobrowiceMw);
	bobrowiceMw = service.mark(bobrowiceMw, bobrowiceRecognitionInfo.getMatchedSegments());
	cv::imshow("Speed limit", bobrowiceMw.getMat());
	cv::waitKey(0);

	RecognitionInfo uszkodzonaJezdniaRecognitionInfo = recognizor.recognize(uszkodzonaJezdniaMw);
	uszkodzonaJezdniaMw = service.mark(uszkodzonaJezdniaMw, uszkodzonaJezdniaRecognitionInfo.getMatchedSegments());
	cv::imshow("Speed limit", uszkodzonaJezdniaMw.getMat());
	cv::waitKey(0);

	RecognitionInfo recognitionInfo30 = recognizor.recognize(mw30);
	mw30 = service.mark(mw30, recognitionInfo30.getMatchedSegments());
	cv::imshow("Speed limit", mw30.getMat());
	cv::waitKey(0);

	RecognitionInfo recognitionInfo40 = recognizor.recognize(mw40);
	mw40 = service.mark(mw40, recognitionInfo40.getMatchedSegments());
	cv::imshow("Speed limit", mw40.getMat());
	cv::waitKey(0);

	RecognitionInfo wiesRecognitionInfo = recognizor.recognize(wiesMw);
	wiesMw = service.mark(wiesMw, wiesRecognitionInfo.getMatchedSegments());
	cv::imshow("Speed limit", wiesMw.getMat());
	cv::waitKey(0);

	RecognitionInfo zimaRecognitionInfo = recognizor.recognize(zimaMw);
	zimaMw = service.mark(zimaMw, zimaRecognitionInfo.getMatchedSegments());
	cv::imshow("Speed limit", zimaMw.getMat());
	cv::waitKey(0);
}
