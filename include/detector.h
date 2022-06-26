#ifndef DEEPLEARNING_DETECTOR_H
#define DEEPLEARNING_DETECTOR_H

#include <string>
#include <opencv2/dnn.hpp>
#include <filesystem>

// Do we want to write predictions to a file
//#define WRITE_MODEL_PREDICTIONS

// Minimum confidence threshold
constexpr float CONFIDENCE_THRESHOLD = 0.6;
// Threshold for non maximum suppresion
constexpr float NMS_THRESHOLD = 0.4;
// Location of model
const std::string ONNX_MODEL = ("./config/model_640.onnx");
constexpr int INPUT_WIDTH = 640;//1280;
constexpr int INPUT_HEIGHT = 640;//1280;
const std::string OUTPUT_DIRECTORY = ("./out/labelled/");

const int DIMENSIONS = 1 + 4; // CONFIDENCE + X,Y,W,H + (NUM DIMENSIONS)
const int ROWS = 25200; //* 4;

struct Detection
{
    int classId{};
    float confidence{};
    cv::Rect box;
};

class Detector
{
public:
    Detector(const std::string& model, const std::vector<std::string>& classNames, const std::vector<cv::Scalar>& colors);
    ~Detector() { };

    static cv::Mat ReformatSource(const cv::Mat& source);

    void GetPredictions(const cv::Mat& input, std::vector<Detection>& output);
    void DrawPredictions(const cv::Mat& source, const std::vector<Detection>& detections);
    void DrawPredictions(const cv::Mat& source, const std::vector<Detection>& detections, cv::Mat (*transform)(const cv::Mat&, const cv::Scalar& color));

    static bool WritePredictions(const cv::Mat& source, const std::vector<Detection>& detections, const std::string& filename);

private:
    cv::dnn::Net _loadedModel;

    std::vector<std::string> _classNames;
    std::vector<cv::Scalar> _colors;
};

#endif //DEEPLEARNING_DETECTOR_H
