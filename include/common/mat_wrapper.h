#include <opencv2/opencv.hpp>

namespace cir {
    namespace common {
        class MatWrapper {
            public:
                cv::Mat getMat();

            private:
                cv::Mat mat;
        };
    }
}
