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
    int scale = 0;
}PipelineConfig_t;

class PipeLine
{
public:
    PipeLine(int scale);
    ~PipeLine();
    int CreatePipeLine(PipelineConfig_t& pipeline_config);
    int Apply(const cv::Mat& input_img, cv::Mat& output_img);

private:
    std::unique_ptr<CodeFormer> codeformer_;
    std::unique_ptr<FaceG> face_detector_;
    PipelineConfig_t pipeline_config_;
};

}  // namespace wsdsb

#endif // PIPELINE_H
