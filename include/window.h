#ifndef DEEPLEARNING_WINDOW_H
#define DEEPLEARNING_WINDOW_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Windows.h>

HWND GetWindowHandle(const char* processName);

class Window
{
public:
    const char* windowName;

    Window(const HWND& window, const char* windowName, int flags);
    Window(const cv::VideoCapture& video, const char* windowName, int flags);

    cv::Mat Update();

    ~Window();

private:
    HWND hwndRef = nullptr;
    cv::VideoCapture capture;
    cv::Mat source;
};

#endif //DEEPLEARNING_WINDOW_H
