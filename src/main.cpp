#include "helpers.h"
#include "pipeline.h"
#include "realesrgan.h"
#include <cstdio>
#include <iostream>
#include <net.h>
#include <string>
#include <string_view>

#include "wic_image.h"

#include <opencv2\core\ocl.hpp>

#define VER "1.01"

#if _WIN32
static wchar_t *optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t *const argv[], const wchar_t *optstring) {
    optarg = NULL;
    if (optind >= argc)
        return 0;

    if (argv[optind][0] != L'-') {
        ++optind;
        return -1;
    }

    wchar_t opt = argv[optind][1];
    const wchar_t *p = wcschr(optstring, opt);
    if (p == NULL) {
        ++optind;
        return L'?';
    } else {
        optarg = argv[optind + 1];
        ++optind;
        return opt;
    }

    optind++;

    return -1;
};
#endif

static void print_usage() {
    fprintf(stderr, "CXVUSER AI MegaPixel XL Super-Black edition Upscale solution " VER ", Welcam...\n\n");
    fprintf(stderr, "This project uses (onnx,ncnn inference) with Vulkan and DirectML compute...\n");
    fprintf(stderr, "Usage: this_binary [options]...\n\n");
    fprintf(stderr, " -i <img> path to image\n");
    fprintf(stderr, " -s <digit> model scale factor (default=autodetect)\n");
    fprintf(stderr, " -j <digit> custom output scale factor\n");
    fprintf(stderr, " -t <digit> tile size (default=auto)\n");
    fprintf(stderr, " -f restore faces (default=GFPGAN ncnn)\n");
    fprintf(stderr, " -m <string> esrgan model name (default=./models/x4nomos8ksc)\n");
    fprintf(stderr, " -g <string> gfpgan model path (default=./models/gfpgan_1.4.onnx)\n");
    fprintf(stderr, " -x <digit> YOLO face detection threshold (default=0,5) (0,3..0,7 recommended)\n");
    fprintf(stderr, " -c use CodeFormer face restore model\n");
    fprintf(stderr, " -d swith face restore infer to onnx\n");
    fprintf(stderr, " -w <digit> CodeFormer Fidelity (Only onnx) (default=0,7)\n");
    fprintf(stderr, " -u Face Upsample (after face restore)\n");
    fprintf(stderr, " -z <string> FaceUpsample model (ESRGAN)\n");
    fprintf(stderr, " -n no upsample\n");
    fprintf(stderr, " -v verbose\n");
};

#if _WIN32
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{
    std::wstring imagepath;
    char imagepatha[_MAX_PATH];

    //default selected models
    std::wstring esr_model = L"./models/4xNomos8kSC";
    char esr_modela[_MAX_PATH] = "./models/4xNomos8kSC";
    std::wstring gfp_model = L"./models/gfpgan_1.4.onnx";
    char gfp_modela[_MAX_PATH] = "./models/gfpgan_1.4.onnx";
    std::wstring fc_up_m;
    char cdf_upa[_MAX_PATH];

    //default processing params
    bool upsample = true;
    int model_scale = 0;
    int custom_scale = 0;
    int tilesize = 20;
    bool restore_face = false;
    bool verbose = false;
    bool use_codeformer = false;
    bool use_infer_onnx = false;
    float codeformer_fidelity = 0.7;
    bool fc_up_ = false;
    float prob_face_thd = 0.5f;
    float nms_face_thd = 0.65f;

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:s:t:j:f:m:g:v:n:h:c:x:w:d:u:z")) != (wchar_t) 0) {
        switch (opt) {
            case L'i': {
                imagepath = optarg;
                wcstombs(imagepatha, imagepath.data(), _MAX_PATH);
            } break;
            case L's': {
                model_scale = _wtoi(optarg);
            } break;
            case L't': {
                tilesize = _wtoi(optarg);
            } break;
            case L'f': {
                restore_face = true;
            } break;
            case L'm': {
                esr_model = optarg;
                wcstombs(esr_modela, esr_model.data(), _MAX_PATH);
            } break;
            case L'g': {
                gfp_model = optarg;
                wcstombs(gfp_modela, gfp_model.data(), _MAX_PATH);
            } break;
            case L'v': {
                verbose = true;
            } break;
            case L'c': {
                use_codeformer = true;
            } break;
            case L'd': {
                use_infer_onnx = true;
            } break;
            case L'w': {
                codeformer_fidelity = _wtof(optarg);
            } break;
            case L'x': {
                prob_face_thd = _wtof(optarg);
            } break;
            case L'n': {
                upsample = false;
            } break;
            case L'u': {
                fc_up_ = true;
            } break;
            case L'z': {
                fc_up_m = optarg;
                wcstombs(cdf_upa, fc_up_m.data(), _MAX_PATH);
            } break;
            case L'j': {
                custom_scale = _wtoi(optarg);
            } break;
            case L'h': {
                print_usage();
                return 0;
            } break;
            case L'?': {
                print_usage();
                return 0;
            } break;
        }
    }
#endif

    if (imagepath.empty() || (false == upsample && false == restore_face)) {
        print_usage();
        return 0;
    }

    unsigned char *pixeldata = 0;
    int w{}, h{}, c{};

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    uint32_t heap_budget = ncnn::get_gpu_device(ncnn::get_default_gpu_index())->get_heap_budget();//VRAM size

    //calculate tilesize for VRAM consumption
    if (heap_budget > 1900)
        tilesize = 200;
    else if (heap_budget > 550)
        tilesize = 100;
    else if (heap_budget > 190)
        tilesize = 64;
    else
        tilesize = 32;

    int haveOpenCL{};
    int useOpenCL{};
    if (haveOpenCL = cv::ocl::haveOpenCL())
        cv::ocl::setUseOpenCL(true);
    useOpenCL = cv::ocl::useOpenCL();

#if _WIN32
    pixeldata = wic_decode_image(imagepath.data(), &w, &h, &c);
    if (!pixeldata) {
        fprintf(stderr, "Failed to load image...\n");
        return 0;
    }
#else
#endif

    if (upsample)
        if (!model_scale) {
            //heuristic model scale detection method
            if (esr_model.find(L"1x", 0) != std::string::npos || esr_model.find(L"x1", 0) != std::string::npos)
                model_scale = 1;
            if (esr_model.find(L"2x", 0) != std::string::npos || esr_model.find(L"x2", 0) != std::string::npos)
                model_scale = 2;
            if (esr_model.find(L"3x", 0) != std::string::npos || esr_model.find(L"x3", 0) != std::string::npos)
                model_scale = 3;
            if (esr_model.find(L"4x", 0) != std::string::npos || esr_model.find(L"x4", 0) != std::string::npos)
                model_scale = 4;
            if (esr_model.find(L"5x", 0) != std::string::npos || esr_model.find(L"x5", 0) != std::string::npos)
                model_scale = 5;
            if (esr_model.find(L"8x", 0) != std::string::npos || esr_model.find(L"x8", 0) != std::string::npos)
                model_scale = 8;
            if (esr_model.find(L"16x", 0) != std::string::npos || esr_model.find(L"x16", 0) != std::string::npos)
                model_scale = 16;

            if (!model_scale) {
                fprintf(stderr, "Error autodetect scale of this upscale model please override scale factor using -s flag for using this model");
                ncnn::destroy_gpu_instance();
                return 0;
            }
        }

    if (true == verbose) {
        fprintf(stderr, "Input image dimensions w: %d, h: %d, c: %d...\n", w, h, c);
        if (custom_scale)
            fprintf(stderr, "Output image dimensions w: %d, h: %d, c: %d...\n", w * custom_scale, h * custom_scale, c);
        else
            fprintf(stderr, "Output image dimensions w: %d, h: %d, c: %d...\n", w * model_scale, h * model_scale, c);

        fprintf(stderr, "tilesize: %d, ncnn_inf: %d, onnx_inf: %d, restore_face: %d,"
                        " model_scale: %d, upsample: %d, use_codeformer: %d\n"
                        " gfp_model_path: %s\n"
                        " esr_model_path: %s\n"
                        " heap_vram_budget: %d\n"
                        " custom_scale: %d\n"
                        " codeformer face upsample: %d\n"
                        " codeformer fidelity: %.2f\n"
                        " face detect threshold: %.2f\n"
                        " OpenCV have OpenCL: %d\n"
                        " OpenCV uses OpenCL: %d\n",
                tilesize, use_infer_onnx ? 0 : 1, use_infer_onnx, restore_face, model_scale,
                upsample, use_codeformer, gfp_modela, esr_modela, heap_budget,
                custom_scale, fc_up_, codeformer_fidelity, prob_face_thd, haveOpenCL, useOpenCL);
    }
    ncnn::Mat bg_upsample_ncnn(w * model_scale, h * model_scale, (size_t) c, c);
    ncnn::Mat bg_presample_ncnn(w, h, (void *) pixeldata, (size_t) c, c);
    cv::Mat bg_upsample_ocv;
    cv::Mat bg_presample_ocv(h, w, (c == 3) ? CV_8UC3 : CV_8UC4, pixeldata);

    std::wstringstream str;
    char stra[_MAX_PATH];

    //------------------------------------- upsampling image -------------------------------------
    {
        if (upsample) {
            RealESRGAN real_esrgan;

            real_esrgan.scale = model_scale;
            real_esrgan.prepadding = 10;
            real_esrgan.tilesize = tilesize;

            std::wstringstream str_param;
            str_param << esr_model.data() << ".param" << std::ends;
            std::wstringstream str_bin;
            str_bin << esr_model.data() << ".bin" << std::ends;

            fprintf(stderr, "Loading upscayl model from %s...\n", esr_modela);
            real_esrgan.load(str_param.view().data(), str_bin.view().data());
            fprintf(stderr, "Loading upscayl model finished...\n");

            fprintf(stderr, "Upscale image...\n");
            real_esrgan.process(bg_presample_ncnn, bg_upsample_ncnn);
            fprintf(stderr, "Upscale image finished...\n");

            str << imagepath << L"_" << getfilew(esr_model.data()) << L"_ms" << model_scale << L"_cs" << custom_scale << ".png" << std::ends;

#if _WIN32
            wic_encode_image(str.view().data(), bg_upsample_ncnn.w, bg_upsample_ncnn.h, bg_upsample_ncnn.elempack, bg_upsample_ncnn.data);
#else
#endif
            wcstombs(stra, str.view().data(), _MAX_PATH);
            if (custom_scale) {
                cv::Mat pre = cv::imread(stra, 1);
                cv::Mat up;
                cv::resize(pre, up, cv::Size(bg_presample_ocv.cols * custom_scale, bg_presample_ocv.rows * custom_scale), 0, 0, cv::InterpolationFlags::INTER_LINEAR);
                cv::imwrite(stra, up);
            }
        } else {
            if (custom_scale)
                cv::resize(bg_presample_ocv, bg_upsample_ocv, cv::Size(bg_presample_ocv.cols * custom_scale, bg_presample_ocv.rows * custom_scale), 0, 0, cv::InterpolationFlags::INTER_LINEAR);
            else
                bg_presample_ocv.copyTo(bg_upsample_ocv);
            str << imagepath << L"_" << getfilew(gfp_model.data()) << L"_s" << model_scale << L"_cs" << custom_scale << "_interpolated"
                << ".png" << std::ends;

#if _WIN32
            wic_encode_image(str.view().data(), bg_upsample_ocv.cols, bg_upsample_ocv.rows, bg_upsample_ocv.elemSize(), bg_upsample_ocv.data);
#else
#endif
        }
        wcstombs(stra, str.view().data(), _MAX_PATH);
    }//------------------------------------- upsampling image -------------------------------------

    {//------------------------------------- Face restore -------------------------------------
        if (true == restore_face) {
            char path[_MAX_PATH];
            wcstombs(path, str.view().data(), _MAX_PATH);
            cv::Mat img_faces_upsamle = cv::imread(path, 1);

            wsdsb::PipelineConfig_t pipeline_config_t;
            pipeline_config_t.model_path = "./models/";
            if (use_infer_onnx)
                pipeline_config_t.onnx = true;
            else
                pipeline_config_t.ncnn = true;

            pipeline_config_t.face_upsample = fc_up_;
            pipeline_config_t.prob_thr = prob_face_thd;

            pipeline_config_t.custom_scale = custom_scale;
            pipeline_config_t.model_scale = model_scale;

            pipeline_config_t.w = codeformer_fidelity;
            strcpy_s(pipeline_config_t.name, 255, stra);
            pipeline_config_t.fc_up_model = fc_up_m;

            if (use_codeformer)
                pipeline_config_t.codeformer = true;
            else
                pipeline_config_t.face_model = gfp_modela;

            wsdsb::PipeLine pipe;
            pipe.CreatePipeLine(pipeline_config_t);
            pipe.Apply(bg_presample_ocv, img_faces_upsamle);

            //------------------------------------- save result image -------------------------------------
            {
                std::stringstream file;
                file << imagepatha;
                if (upsample)
                    file << "_" << getfilea(esr_modela);
                if (restore_face) {
                    if (use_codeformer) {
                        file << "_codeformer";
                        if (use_infer_onnx) {
                            file << "_w" << codeformer_fidelity;
                            if (fc_up_)
                                file << "_fu_" << getfilea(cdf_upa);
                        }
                    } else {
                        file << "_" << getfilea(gfp_modela);
                    }
                }

                file << "_ms" << model_scale;
                if (custom_scale)
                    file << "_cs" << custom_scale;

                file << ".png" << std::ends;

                cv::imwrite(file.view().data(), img_faces_upsamle);
            }
            //------------------------------------- Save result image -------------------------------------
        }//------------------------------------- Face restore -------------------------------------
    }
#if _WIN32
    CoUninitialize();
#endif

    ncnn::destroy_gpu_instance();

    fprintf(stderr, "Finish enjoy...\n");

    return 0;
};