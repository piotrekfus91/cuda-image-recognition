#include "cir/common/recognition/Recognizor.h"
#include <boost/chrono.hpp>

using namespace cir::common::logger;

namespace cir { namespace common { namespace recognition {

Recognizor::Recognizor(ImageProcessingService& service) : _service(service) {

}

Recognizor::~Recognizor() {

}

const RecognitionInfo Recognizor::recognize(MatWrapper& input) {
	boost::chrono::high_resolution_clock::time_point start = boost::chrono::high_resolution_clock::now();

	const RecognitionInfo recognitionInfo = doRecognize(input);

	boost::chrono::high_resolution_clock::time_point end = boost::chrono::high_resolution_clock::now();
	boost::chrono::nanoseconds totalTimeInNano = end - start;
	double totalTimeInSec = totalTimeInNano.count() / 1000000000.;

	Logger* logger = _service.getLogger();
	logger->log("Recognition", totalTimeInSec);

	return recognitionInfo;
}

}}}
