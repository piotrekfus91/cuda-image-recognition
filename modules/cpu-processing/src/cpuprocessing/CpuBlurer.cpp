#include "cir/cpuprocessing/CpuBlurer.h"

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuBlurer::CpuBlurer() {

}

CpuBlurer::~CpuBlurer() {

}

template<class T>
void sort(T* arr, int size) {
	for(int i = 0; i < size; i++) {
		for(int j = i + 1; j < size; j++) {
			T t1 = arr[i];
			T t2 = arr[j];
			if(t2 < t1) {
				arr[i] = t2;
				arr[j] = t1;
			}
		}
	}
}

MatWrapper CpuBlurer::doMedian(const MatWrapper& mw, int size) {
	cv::Mat mat = mw.getMat();
	cv::Mat clone = mat.clone();
	int width = mat.cols;
	int height = mat.rows;
	uchar* data = mat.data;
	uchar* cloneData = clone.data;

	uchar surround[(2*size+1) * (2*size+1)];
	int total;
	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			total = 0;
			for(int i = x - size; i <= x + size; i++) {
				for(int j = y - size; j <= y + size; j++) {
					if(i >= 0 && i < width && j >= 0 && j < height) {
						surround[total++] = data[j * width + i];
					}
				}
			}
			sort<uchar>(surround, total);
			uchar u = total % 2 == 0 ? (surround[total / 2] + surround[total / 2 - 1]) / 2 : surround[total / 2 - 1];
			cloneData[(x + width * y)] = u;
		}
	}

	MatWrapper outputMw(clone);
	outputMw.setColorScheme(mw.getColorScheme());
	return outputMw;
}

}}
