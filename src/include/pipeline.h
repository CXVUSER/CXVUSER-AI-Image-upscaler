#ifndef PIPELINE_H
#define PIPELINE_H

#include "codeformer.h"
#include "gfpgan.h"
#include "include/Faceyolov7_lite_e.h"
#include "include/retinaface.h"
#include "include/Faceyolov5bl.h"
#include "include/helpers.h"
#include "realesrgan.h"

typedef struct _PipelineConfig {
    bool bg_upsample = false;
    bool face_upsample = false;
    float w = 0.7;
    std::string model_path;
    std::wstring fc_up_model;
    std::string face_model;
    std::wstring face_det_model;
    int custom_scale = 0;
    int model_scale = 0;
    bool ncnn = false;
    bool onnx = false;
    char name[_MAX_PATH];
    wchar_t namew[_MAX_PATH];
    float prob_thr = 0.5f;
    float nms_thr = 0.65f;
    bool codeformer = false;
    bool useParse = false;

} PipelineConfig_t;

//class onnxContext {
//public:
//    void init(const std::filesystem::path &modelPath);
//    static void RunModel(
//            const std::filesystem::path &imagePath,
//            PipelineConfig_t &pipe);
//    static void RunCodeformerModel(
//            const std::filesystem::path &imagePath,
//            PipelineConfig_t &pipe);
//
//private:
//    Microsoft::WRL::ComPtr<IDMLDevice> idl;
//    Microsoft::WRL::ComPtr<ID3D12CommandQueue> d3dQueue;
//    const OrtDmlApi *ortDmlApi = nullptr;
//    Ort::Session* ortSession;
//    Ort::Env* env;
//};

class PipeLine {
public:
    PipeLine();
    ~PipeLine();
    int CreatePipeLine(PipelineConfig_t &pipeline_config);
    int Apply(const cv::Mat &input_img, cv::Mat &output_img);
    void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample);

protected:
    void RunModel(
            const std::filesystem::path &modelPath,
            const std::filesystem::path &imagePath);

private:
    CodeFormer codeformer_NCNN_;
    FaceDetModel *face_detector;
    GFPGAN gfpgan_NCNN_;//GFPGANCleanv1-NoCE-C2
    RealESRGAN face_up_NCNN_;
    PipelineConfig_t pipe;
    ncnn::Net parsing_net;
};
#endif// PIPELINE_H