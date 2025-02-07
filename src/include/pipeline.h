#ifndef PIPELINE_H
#define PIPELINE_H

#include "gfpgan.h"
#include "realesrgan.h"
#include "codeformer.h"
#include "include/face.h"
#include "include/helpers.h"

namespace wsdsb{
typedef struct _PipelineConfig {
    bool bg_upsample = false;
    bool face_upsample = false;
    float w = 0.7;
    std::string model_path;
    std::wstring fc_up_model;
    std::string face_model;
    int custom_scale = 0;
    int model_scale = 0;
    bool ncnn = false;
    bool onnx = false;
    char name[_MAX_PATH];
    wchar_t namew[_MAX_PATH];
    float prob_thr = 0.5f;
    float nms_thr = 0.65f;
    bool codeformer = false;

}PipelineConfig_t;

class PipeLine
{
public:
    PipeLine();
    ~PipeLine();
    int CreatePipeLine(PipelineConfig_t& pipeline_config);
    int Apply(const cv::Mat& input_img, cv::Mat& output_img);
    void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample, PipelineConfig_t &pipe);

private:
    CodeFormer* codeformer_NCNN_;
    FaceG* face_detector_NCNN_;
    GFPGAN* gfpgan_NCNN_; //GFPGANCleanv1-NoCE-C2
    RealESRGAN* face_up_NCNN_;
    PipelineConfig_t pipeline_config_;
};

}  // namespace wsdsb

#endif // PIPELINE_H
