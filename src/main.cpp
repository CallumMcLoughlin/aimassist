#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <ctime>

#include "detector.h"
#include "transform.h"
#include "window.h"

// Quit key
#define ESC 27

/**
 * Read from process
 *
 * @param processName (partial) name of process
 * @return Window pointer
 */
std::unique_ptr<Window> GetNewWindow(const char* processName)
{
    HWND handle = GetWindowHandle(processName);
    if (!handle) {
        return nullptr;
    }

    auto window = std::make_unique<Window>(handle, "Output", cv::WINDOW_NORMAL);
    cv::resizeWindow(window->windowName, 1920 / 2, 1080 / 2);

    return window;
}

/**
 * Debug read from webcam or set to read from video file
 */
std::unique_ptr<Window> GetNewWindow()
{
    cv::VideoCapture video(0);
    auto window = std::make_unique<Window>(video, "Output", cv::WINDOW_NORMAL);
    cv::resizeWindow(window->windowName, 1920 / 2, 1080 / 2);

    return window;
}

/**
 * Convert ticks to milliseconds
 */
double clockToMilliseconds(clock_t ticks){
    // units/(units/time) => time (seconds) * 1000 = milliseconds
    return (ticks/(double)CLOCKS_PER_SEC)*1000.0;
}

/**
 * Main program entry
 */
int main(int argc, char** argv)
{
    if (argc <= 1) {
        std::cerr << "Supply (part of) a case sensitive process name to read from" << std::endl;
        return 1;
    }

    std::unique_ptr<Window> window = GetNewWindow(argv[1]);
    if (!window) {
        std::cerr << "Failed to find window: " << argv[1] << std::endl;
        return 1;
    }

    std::vector<std::string> classNames = { "CT", "T" };
    std::vector<cv::Scalar> colors = {cv::Scalar(255, 111, 0), cv::Scalar(50, 0, 255)};
    auto detector = Detector { ONNX_MODEL, classNames, colors };

    int key = 0;
    clock_t deltaTime = 0;
    unsigned int frames = 0;
    double  frameRate = 30;
    double  averageFrameTimeMilliseconds = 33.333;

    // Main body loop
    while(key != ESC)
    {
        // Framerate calculation
        clock_t beginFrame = clock();

        // Get next frame
        cv::Mat source = window->Update();
        if (source.empty()) {
            // Out of frames
            break;
        }

        cv::Mat3b newSource;
        // Convert BGRA -> BGR
        cv::cvtColor(source, newSource, cv::COLOR_BGRA2BGR);

        // Store what we detect
        std::vector<Detection> output;
        detector.GetPredictions(newSource, output);
#ifdef WRITE_MODEL_PREDICTIONS
        // Optionally write out these detections to a file
        Detector::WritePredictions(newSource, output, std::to_string(count));
#endif
        // Draw predictions on top of input
        detector.DrawPredictions(newSource, output, Transform::DrawOutline);

        // Show to user
        cv::imshow(window->windowName, newSource);

        // Calculate framerate
        clock_t endFrame = clock();
        deltaTime += endFrame - beginFrame;
        frames++;

        // Print out every second
        if(clockToMilliseconds(deltaTime) > 1000.0){
            frameRate = (double)frames * 0.5 + frameRate * 0.5;
            frames = 0;
            deltaTime -= CLOCKS_PER_SEC;
            averageFrameTimeMilliseconds  = 1000.0/(frameRate == 0 ? 0.001 : frameRate);

            std::cout << averageFrameTimeMilliseconds << "," << frameRate << std::endl;
        }

        key = cv::waitKey(1);
    }
    return 0;
}