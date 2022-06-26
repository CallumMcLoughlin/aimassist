#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Windows.h>
#include <iostream>

#include "window.h"

struct HandleContainer {
    LPARAM lparam;
    const char* str;
    HWND window;
};

/**
 * Windows API, get process handle
 */
BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam)
{
    HandleContainer& container = *reinterpret_cast<HandleContainer*>(lParam);

    char buffer[128];
    int written = GetWindowTextA(hwnd, buffer, 128);

    if (written && strstr(buffer, container.str) != nullptr) {
        container.window = hwnd;
        return FALSE;
    }

    return TRUE;
}

/**
 * Get windows handle from process name
 */
HWND GetWindowHandle(const char* processName)
{
    HandleContainer container {
        .lparam = 0,
        .str = processName,
        .window = nullptr,
    };

    // Result is in HandleContainer struct
    EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&container));

    return container.window;
}

/**
 * Turn window handle into opencv mat
 */
cv::Mat Hwnd2Mat(HWND hwnd)
{
    HDC hwindowDC;
    HDC hwindowCompatibleDC;

    int height;
    int width;
    int srcheight;
    int srcwidth;

    HBITMAP hbwindow;
    cv::Mat src;

    BITMAPINFOHEADER bi;

    hwindowDC = GetDC(hwnd);
    hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
    SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

    RECT windowsize;
    GetClientRect(hwnd, &windowsize);

    srcheight = windowsize.bottom;
    srcwidth = windowsize.right;
    height = windowsize.bottom;
    width = windowsize.right;

    src.create(height, width, CV_8UC4);

    hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;

    SelectObject(hwindowCompatibleDC, hbwindow);
    StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY);
    GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    DeleteObject(hbwindow);
    DeleteDC(hwindowCompatibleDC);
    ReleaseDC(hwnd, hwindowDC);

    return src;
}

Window::Window(const HWND& window, const char* windowName, int flags) {
    this->windowName = windowName;
    hwndRef = window;
    cv::namedWindow(windowName, flags);
}

Window::Window(const cv::VideoCapture& video, const char *windowName, int flags) {
    this->windowName = windowName;
    capture = video;
    cv::namedWindow(windowName, flags);
}

/**
 * Get next frame from window or from videocapture source
 */
cv::Mat Window::Update() {
    if (hwndRef != nullptr) {
        source = Hwnd2Mat(hwndRef);
    } else {
        int cameraCheckIterations = 20;
        do
        {
            capture >> source;
            cameraCheckIterations--;
        } while (source.empty() && cameraCheckIterations > 0);
        // If we have some invalid frames, allow some skipping before aborting entirely
    }

    return source;
}

/**
 * Release capture window
 */
Window::~Window() {
    if (capture.isOpened()) {
        capture.release();
    }
}
