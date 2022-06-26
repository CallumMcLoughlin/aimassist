#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "transform.h"

/**
 * Convert and return source in gray format
 */
cv::Mat Transform::ToGrey(const cv::Mat& source) {
    cv::Mat grey_source;
    cv::cvtColor(source, grey_source, cv::COLOR_BGR2GRAY);
    return grey_source;
}

/**
 * Calculate and draw outline onto source
 * @param source Source (gray) image
 * @param color Colour of outline
 * @return Drawn ontop mat
 */
cv::Mat Transform::DrawOutline(const cv::Mat& source, const cv::Scalar& color) {
    cv::Mat output = source;
    auto greySource = Transform::ToGrey(source);

    // Gaussian blur
    cv::GaussianBlur(greySource, greySource, cv::Size(7, 7), 1, 1);

    // Closing + Morphological Gradient
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
    cv::morphologyEx(greySource, greySource, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 2);
    cv::morphologyEx(greySource, greySource, cv::MORPH_GRADIENT, element, cv::Point(-1, -1), 1);

    // Threshold + Otsu's method for threshold params
    cv::threshold(greySource, greySource, 255, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Draw outlines based on threshhold mask
    for (int i = 0; i < greySource.rows; i++) {
        for (int j = 0; j < greySource.cols; j++) {
            if (greySource.at<uchar>(i, j) == 255)
            {
                output.at<cv::Vec3b>(i, j)[0] = color[0];
                output.at<cv::Vec3b>(i, j)[1] = color[1];
                output.at<cv::Vec3b>(i, j)[2] = color[2];
            }
        }
    }

    return output;
}