#ifndef DEEPLEARNING_TRANSFORM_H
#define DEEPLEARNING_TRANSFORM_H

#include <opencv2/core/mat.hpp>

class Transform
{
public:
    static cv::Mat DrawOutline(const cv::Mat& source, const cv::Scalar& color);
    // Add more transforms with same signature as above

private:
    static cv::Mat ToGrey(const cv::Mat& source);
};

#endif //DEEPLEARNING_TRANSFORM_H