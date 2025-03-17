#ifndef FACE_YOLOV5_BL
#define FACE_YOLOV5_BL
#include "include/model.h"
#include "net.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

struct Object {
    cv::Rect_<float> rect;
    int label;
    float score;
    std::vector<cv::Point2f> pts;
    cv::Mat trans_inv;
};

class Face_yolov5_bl : public FaceDetModel {
public:
    Face_yolov5_bl();
    ~Face_yolov5_bl();
    int Load(const std::wstring &model_path) override;
    int Process(const cv::Mat &input_img, void *result) override;
    int align_warp_face(const cv::Mat &img, Object_t &objects);
    void draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects);
    void setScale(int scale_);
    void setThreshold(float prob_threshold_, float nms_threshold_);

protected:
    void Run(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor) override;
    void PreProcess(const void *input_data, std::vector<Tensor_t> &input_tensor) override;
    void PostProcess(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor, void *result) override;

private:
    ncnn::Net net;
    float prob_threshold;
    float nms_threshold;
    int scale;
};

#endif// FACE_YOLOV5_BL
