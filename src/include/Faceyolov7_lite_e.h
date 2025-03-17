#ifndef FACEYOLOV7_LITE_E
#define FACEYOLOV7_LITE_E
#include "include/model.h"
#include <cmath>

class Faceyolov7_lite_e : public FaceDetModel {
public:
    Faceyolov7_lite_e();
    ~Faceyolov7_lite_e();
    int Load(const std::wstring &model_path) override;
    int Process(const cv::Mat &input_img, void *result) override;
    void setScale(int scale_);
    void setThreshold(float prob_threshold_, float nms_threshold_);

protected:
    void Run(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor) override;
    void PreProcess(const void *input_data, std::vector<Tensor_t> &input_tensor) override;
    void PostProcess(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor, void *result) override;

private:
    void AlignFace(const cv::Mat &img, Object_t &objects);
    void draw_objects(const cv::Mat &bgr, const std::vector<Object_t> &objects);

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
#endif// FACEYOLOV7_LITE_E
