#include <locale.h>
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>
#include "cir/common/classification/TesseractClassifier.h"
#include "cir/common/exception/OcrException.h"

using namespace cir::common;
using namespace cir::common::exception;
using namespace cir::common::recognition;
using namespace std;

namespace cir { namespace common { namespace classification {

TesseractClassifier::TesseractClassifier() : _boundary(2) {
	_tesseract = new tesseract::TessBaseAPI();
	setlocale(LC_NUMERIC, "en_GB.utf8");
	if(_tesseract->Init(NULL, "eng")) {
		throw new OcrException();
	}
}

TesseractClassifier::~TesseractClassifier() {
	if(_tesseract != NULL) {
		delete _tesseract;
	}
}

bool TesseractClassifier::singleChar() const {
	return true;
}

string TesseractClassifier::detect(MatWrapper& input, ImageProcessingService* service,
		const map<string, Pattern*>* patternsMap, Segment* segment) {
	MatWrapper segmentMw = input.clone();

	if(segment != NULL) {
		if(segment->leftX - _boundary >= 0)
			segment->leftX -= _boundary;

		if(segment->rightX + _boundary < input.getWidth())
			segment->rightX += _boundary;

		if(segment->topY - _boundary >= 0)
			segment->topY -= _boundary;

		if(segment->bottomY + _boundary < input.getHeight())
			segment->bottomY += _boundary;

		segmentMw = service->crop(segmentMw, segment);
	}

	cv::Mat mat = segmentMw.getMat();
	_tesseract->SetImage(mat.data, mat.cols, mat.rows, mat.channels(), mat.step);
	char* detectedSigns = _tesseract->GetUTF8Text();
	string result;
	for(unsigned int i = 0; i < strlen(detectedSigns); i++) {
		char c = detectedSigns[i];
		if(('A' <= c && c <= 'Z') || ('0' <= c && c <= '9')) {
			result.push_back(c);
			break;
		}
	}
	return result;
}

}}}
