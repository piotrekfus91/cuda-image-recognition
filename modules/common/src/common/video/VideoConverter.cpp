#include "cir/common/video/VideoConverter.h"

using namespace cir::common;

namespace cir { namespace common { namespace video {

VideoConverter::VideoConverter(ImageProcessingService* service) : _service(service) {

}

VideoConverter::~VideoConverter() {

}

ImageProcessingService* VideoConverter::getService() {
	return _service;
}

}}}
