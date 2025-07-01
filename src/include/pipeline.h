#ifndef PIPELINE_H
#define PIPELINE_H

#include "codeformer.h"
#include "gfpgan.h"
#include "include/ColorSiggraph.h"
#include "include/Faceyolov5bl.h"
#include "include/Faceyolov7_lite_e.h"
#if defined(_WIN32)
#if defined(USE_DM)
#include "include/helpers.h"
#endif
#endif
#include "include/retinaface.h"
#include "realesrgan.h"
#include <opencv2\core\ocl.hpp>

typedef struct _PipelineConfig {
    bool bg_upsample = false;
    bool tta_mode = false;
    bool twox_mode = false;
    bool face_upsample = false;
    bool face_restore = true;
    float w = 0.7;
    std::wstring esr_model;     // ESRGAN model path
    std::wstring model_path;    // root model path
    std::wstring fc_up_model;   // face upsample model
    std::wstring face_model;    // gfp model
    std::wstring face_det_model;// face detection model
    std::wstring colorize_m;    // Color restore model
    int custom_scale = 0;
    int model_scale = 0;
    bool onnx = false;
    float prob_thr = 0.5f;
    float nms_thr = 0.65f;
    bool codeformer = false;
    bool useParse = false;
    int Colorize = false;
    bool gpu = true;
    int tilesize = 0;
} PipelineConfig_t;

#define CLASS_EXPORT class __declspec(dllexport)

CLASS_EXPORT PipeLine {
public:
    PipeLine();
    ~PipeLine();
    int LaunchEngine(PipelineConfig_t & pipeline_config);

    cv::Mat Apply(const cv::Mat &input_img);
    static PipeLine *getApi();
    int getModelScale(std::wstring str_bins);
    int getEffectiveTilesize();
    std::vector<cv::Mat> &getCrops();
    int load_bg_esr_model(PipelineConfig_t & cfg);
    int load_face_model(PipelineConfig_t & cfg);
    int load_face_up_model(PipelineConfig_t & cfg);
    int load_face_det_model(PipelineConfig_t & cfg);
    int changeScaleFactor(PipelineConfig_t & cfg);
    int changeCodeformerFiledily(PipelineConfig_t & cfg);
    int setFaceDetectorThreshold(PipelineConfig_t & cfg);
    int setUseParse(PipelineConfig_t & cfg);
    int switchToNCNNFaceModels(PipelineConfig_t & cfg);
    int load_color_model(PipelineConfig_t & cfg);
    int changeColorState(PipelineConfig_t & cfg);
    int setESRTTAand2x(PipelineConfig_t & cfg);

private:
    cv::Mat inferFaceModel(
            const cv::Mat &input_img);
    void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample);
    void Clear();
    CodeFormer *codeformer_ncnn = nullptr;
    FaceDetModel *face_detector = nullptr;
    GFPGAN *gfpgan_ncnn = nullptr;
    RealESRGAN *face_upsampler = nullptr;
    RealESRGAN *bg_upsampler = nullptr;
    PipelineConfig_t pipe;
    std::vector<cv::Mat> crops;
    ncnn::Net *parsing_net = nullptr;
    ColorSiggraph *color = nullptr;

    //ONNX
    Ort::SessionOptions *sessionOptions = nullptr;
    const OrtApi &ortApi = Ort::GetApi();
    Ort::Env *env = nullptr;
    Ort::Session *ortSession = nullptr;

#if defined(_WIN32)
#if defined(USE_DM)
    std::tuple<Microsoft::WRL::ComPtr<IDMLDevice>, Microsoft::WRL::ComPtr<ID3D12CommandQueue>> dml;
    const OrtDmlApi *ortDmlApi = nullptr;
#endif
#endif
};
#endif// PIPELINE_H