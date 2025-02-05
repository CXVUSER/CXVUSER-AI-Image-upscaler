#ifndef PIPELINE_H
#define PIPELINE_H

#include "codeformer.h"
#include "include/face2.h"

namespace wsdsb{
typedef struct _PipelineConfig {
    bool bg_upsample = false;
    bool face_upsample = false;
    float w = 0.7;
    std::string model_path;
    std::wstring up_model;
    int custom_scale = 0;
    int model_scale = 0;
    bool ncnn = false;
    bool onnx = false;
    char name[_MAX_PATH];
    wchar_t namew[_MAX_PATH];
    float prob_thr = 0.5f;
    float nms_thr = 0.65f;
}PipelineConfig_t;

class PipeLine
{
public:
    PipeLine();
    ~PipeLine();
    int CreatePipeLine(PipelineConfig_t& pipeline_config);
    int Apply(const cv::Mat& input_img, cv::Mat& output_img);

private:
    CodeFormer* codeformer_;
    FaceG* face_detector_;
    PipelineConfig_t pipeline_config_;
};

}  // namespace wsdsb

#endif // PIPELINE_H
