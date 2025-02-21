#ifndef PIPELINE_H
#define PIPELINE_H

#include "codeformer.h"
#include "gfpgan.h"
#include "include/Faceyolov5bl.h"
#include "include/Faceyolov7_lite_e.h"
#include "include/helpers.h"
#include "include/ColorSiggraph.h"
#include "include/retinaface.h"
#include "realesrgan.h"
#include <opencv2\core\ocl.hpp>

typedef struct _PipelineConfig {
    bool bg_upsample = false;
    bool face_upsample = false;
    bool face_restore = true;
    float w = 0.7;
    std::wstring esr_model; //ESRGAN model path
    std::wstring model_path; //root model path
    std::wstring fc_up_model; //face upsample model
    std::wstring face_model; //gfp model
    std::wstring face_det_model; //face detection model
    int custom_scale = 0;
    int model_scale = 0;
    bool onnx = false;
    float prob_thr = 0.5f;
    float nms_thr = 0.65f;
    bool codeformer = false;
    bool useParse = false;
    bool colorize = false;
    bool gpu = true;
} PipelineConfig_t;

enum AI_SettingsOp {
    //ESRGAN model
    //supports ncnn *.bin:*.param files
    //.esr_model in PipelineConfig_t
    //relative or absolute paths without extensions
    CHANGE_ESR = 1,
    
    //Face restore model
    //supports onnx *.onnx files
    //.face_model in PipelineConfig_t
    //relative or absolute paths without extensions
    CHANGE_GFP,

    //Face detect model
    //supports only ncnn models
    //.face_det_model in PipelineConfig_t
    //(y7 = yolov7_lite_e|y5 = yolov5blazeface|r50 = retinaface)
    CHANGE_FACE_DET,

    //ESRGAN face upsample model
    //supports ncnn *.bin:*.param files
    //.esr_model in PipelineConfig_t
    //relative or absolute paths without extensions
    CHANGE_FACE_UP_M,

    //.custom_scale in PipelineConfig_t
    CHANGE_SCALE_FACTOR,

    //.w in PipelineConfig_t
    CHANGE_CODEFORMER_FID,

    //.prob_thr and .nms_thr in PipelineConfig_t
    CHANGE_FACEDECT_THD,

    //Use face parse model
    //.useParse in PipelineConfig_t
    CHANGE_FACE_PARSE,

    //Change inver (onnx,ncnn)
    //.ncnn .onnx .codeformer in PipelineConfig_t
    CHANGE_INFER,

    //Change colorization state
    //.colorize in PipelineConfig_t
    CHANGE_COLOR
};

#if defined(AS_DLL)
#define CLASS_EXPORT class __declspec(dllexport)
#else
#define CLASS_EXPORT class
#endif

CLASS_EXPORT PipeLine {
public:
    PipeLine();
    ~PipeLine();
    int CreatePipeLine(PipelineConfig_t &pipeline_config);
    
    cv::Mat Apply(const cv::Mat &input_img);
    static PipeLine *getApi();
    void changeSettings(int type, PipelineConfig_t &cfg);
    int getModelScale(std::wstring str_bins);
    int getEffectiveTilesize();
    std::vector<cv::Mat>& getCrops();

private:
    cv::Mat inferONNXModel(
            const cv::Mat &input_img);
    void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample);
    void Clear();
    CodeFormer* codeformer_NCNN_ = nullptr;
    FaceDetModel *face_detector = nullptr;
    GFPGAN* gfpgan_NCNN_ = nullptr;//GFPGANCleanv1-NoCE-C2
    RealESRGAN* face_up_NCNN_ = nullptr;
    RealESRGAN* bg_upsample_md = nullptr;
    PipelineConfig_t pipe;
    std::vector<cv::Mat> crops;
    ncnn::Net parsing_net;
    ColorSiggraph *color = nullptr;

    //ONNX
    Ort::SessionOptions sessionOptions;
    const OrtApi &ortApi = Ort::GetApi();
    Ort::Env *env = nullptr;
    Ort::Session *ortSession = nullptr;

#if defined(_WIN32)
    std::tuple<Microsoft::WRL::ComPtr<IDMLDevice>, Microsoft::WRL::ComPtr<ID3D12CommandQueue>> dml;
    const OrtDmlApi *ortDmlApi = nullptr;
#endif
};
#endif// PIPELINE_H