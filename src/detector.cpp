#include <opencv2/opencv.hpp>
#include <fstream>

#include "detector.h"

/**
 * Load model from file
 * @param model Model location
 * @param classNames Class names
 * @param colors Class colors
 */
Detector::Detector(const std::string& model, const std::vector<std::string>& classNames, const std::vector<cv::Scalar>& colors)
{
    _loadedModel = cv::dnn::readNet(model);
    _loadedModel.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    _loadedModel.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    _classNames = classNames;
    _colors = colors;

    // If we want to write our predictions to a file
#ifdef WRITE_MODEL_PREDICTIONS
    std::filesystem::create_directory(OUTPUT_DIRECTORY);
    std::filesystem::create_directory(OUTPUT_DIRECTORY + "labels");
    std::filesystem::create_directory(OUTPUT_DIRECTORY + "images");
    std::ofstream labelMap(OUTPUT_DIRECTORY + "labelled.labels");
    for (const auto& name : classNames) {
        labelMap << name << std::endl;
    }
#endif

}

/**
 * Convert input into writeable source (CV_8UC3)
 * @param source Input source, possibly in CV_8UC4 form
 * @return Correctly formatted source
 */
cv::Mat Detector::ReformatSource(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

/**
 * Get predictions from DNN model
 * @param input Input frame
 * @param output Output detections
 */
void Detector::GetPredictions(const cv::Mat& input, std::vector<Detection>& output)
{
    auto inputImage = Detector::ReformatSource(input);

    cv::Mat blob;
    cv::dnn::blobFromImage(inputImage, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    _loadedModel.setInput(blob);
    std::vector<cv::Mat> outputs;
    _loadedModel.forward(outputs, _loadedModel.getUnconnectedOutLayersNames());

    double xFactor = (double)inputImage.cols / INPUT_WIDTH;
    double yFactor = (double)inputImage.rows / INPUT_HEIGHT;

    auto* data = (float*)outputs[0].data;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < ROWS; ++i)
    {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classScores = data + 5;
            cv::Mat scores(1, _classNames.size(), CV_32FC1, classScores);
            cv::Point classId;
            double maxClassScore;
            minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classId);
            if (maxClassScore > CONFIDENCE_THRESHOLD)
            {
                confidences.push_back(confidence);
                classIds.push_back(classId.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * xFactor);
                int top = int((y - 0.5 * h) * yFactor);
                int width = int(w * xFactor);
                int height = int(h * yFactor);

                boxes.emplace_back(left, top, width, height);
            }
        }

        data += DIMENSIONS + _classNames.size();
    }

    // Non maximum suppression
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD * NMS_THRESHOLD, NMS_THRESHOLD, nmsResult);

    // Save each detection
    for (int idx : nmsResult)
    {
        Detection result;
        result.classId = classIds[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

/**
 * Clamp box to valid bounds
 */
void ClampBox(cv::Rect& box, int width, int height)
{
    if (box.x < 0) {
        box.x = 0;
    }
    if (box.y < 0) {
        box.y = 0;
    }

    if (box.width + box.x >= width) {
        box.width = width - box.x;
    }

    if (box.height + box.y >= height) {
        box.height = height - box.y;
    }
}

/**
 * Draw predictions onto source
 */
void Detector::DrawPredictions(const cv::Mat& source, const std::vector<Detection>& detections)
{
    int detectionCount = detections.size();
    for (int i = 0; i < detectionCount; i++)
    {
        auto detection = detections[i];
        auto box = detection.box;
        ClampBox(box, source.cols, source.rows);

        auto classId = detection.classId;
        const auto color = _colors[classId % _colors.size()];

        cv::rectangle(source, box, color, 3);
        cv::putText(source, _classNames[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_TRIPLEX, 2, cv::Scalar(0, 0, 255));
    }
}

/**
 * Draw predictions onto source with transform applied
 */
void Detector::DrawPredictions(const cv::Mat& source, const std::vector<Detection>& detections, cv::Mat (*transform)(const cv::Mat&, const cv::Scalar& color))
{
    int detectionCount = detections.size();
    for (int i = 0; i < detectionCount; i++)
    {
        auto detection = detections[i];
        auto box = detection.box;
        ClampBox(box, source.cols, source.rows);

        auto classId = detection.classId;
        const auto color = _colors[classId % _colors.size()];

        auto region = source(box);

        // Apply image transformation
        auto out = transform(region, color);
        out.copyTo(source(box));
        cv::putText(source, _classNames[classId], cv::Point(box.x, box.y), cv::FONT_HERSHEY_TRIPLEX, 1, color);
    }
}

/**
 * Write predictions to source and label files
 */
bool Detector::WritePredictions(const cv::Mat &source, const std::vector<Detection> &detections, const std::string& filename)
{
    std::ofstream file(OUTPUT_DIRECTORY + "labels\\" + filename + ".txt");
    cv::imwrite(OUTPUT_DIRECTORY + "images\\" + filename + ".png", source);
    for (auto detection : detections) {
        double center_width = (detection.box.x + (detection.box.width / 2.)) / source.cols;
        double center_height = (detection.box.y + (detection.box.height / 2.)) / source.rows;
        file << detection.classId << " " << center_width << " " << center_height << " " << (double)detection.box.width / (double)source.cols << " " << (double)detection.box.height / (double)source.rows << std::endl;
    }

    file.close();
    return true;
}
