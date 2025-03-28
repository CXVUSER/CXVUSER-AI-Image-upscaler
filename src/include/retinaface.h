#ifndef FACER_H
#define FACER_H
#include "include/model.h"
#include <cmath>

struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

class FaceR : public FaceDetModel {
public:
    FaceR(bool gpu = true, bool mnet = false);
    ~FaceR();
    int Load(const std::wstring &model_path) override;
    void setThreshold(float prob_threshold_, float nms_threshold_);
    int Process(const cv::Mat &bgr, void *result) override;

private:
    void AlignFace(const cv::Mat &img, Object_t &objects);
    void draw_faceobjects(const cv::Mat &bgr, const std::vector<FaceObject> &faceobjects);

protected:
    void Run(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor) override;
    void PreProcess(const void *input_data, std::vector<Tensor_t> &input_tensor) override;
    void PostProcess(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor, void *result) override;

private:
    float prob_threshold;
    float nms_threshold;
    const float norm_vals_[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};
    std::vector<cv::Point2f> face_template;
    ncnn::Net net_;
    bool gpu;
    bool mnet;
    std::vector<int> input_indexes_;
    std::vector<int> output_indexes_;
};
#endif// FACER_H
