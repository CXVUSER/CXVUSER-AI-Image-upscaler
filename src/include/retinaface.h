#ifndef FACER_H
#define FACER_H
#include <cmath>
#include "include/model.h"

struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

class FaceR : public FaceDetModel {
public:
    FaceR();
    ~FaceR();
    int Load(const std::string &model_path) override;
    void setScale(int scale_);
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
    int scale;
    std::vector<int> input_indexes_;
    std::vector<int> output_indexes_;
};
#endif// FACER_H
