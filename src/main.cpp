#include "helpers.h"
#include "pipeline.h"
#include "realesrgan.h"
#include "wic_image.h"
#include <conio.h>
#include <cstdio>
#include <iostream>
#include <net.h>
#include <string>
#include <string_view>

#include <opencv2\core\ocl.hpp>

#define VER "2.01"

static wchar_t *optarg = NULL;
static int optind = 1;
static wchar_t parsecmd(int argc, wchar_t *const argv[], const wchar_t *optstring) {
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

static void print_usage() {
    const char *str = "CXVUSER AI MegaPixel XL Super-Black Edition Upscale Solution " VER ", Welcome...\n"
                      "This project uses (onnx, ncnn inference) with Vulkan and DirectML compute...\n\n"
                      "Usage: this_binary [options]...\n\n"
                      " -i <img>      Path to input image\n"
                      " -s <digit>    Model scale factor (default=autodetect)\n"
                      " -j <digit>    Custom output scale factor\n"
                      " -t <digit>    Tile size (default=auto)\n"
                      " -f            Restore faces (default=CodeFormer)\n"
                      " -m <string>   ESRGAN model name (default=./models/ESRGAN/4xNomos8kSC)\n"
                      " -g <string>   GFPGAN model path (default=./models/face_restore/codeformer_0_1_0.onnx)\n"
                      " -x <digit>    Face detection threshold (default=0.5, recommended range: 0.3-0.7)\n"
                      " -c            Use CodeFormer face restore model (ncnn)\n"
                      " -d            Switch face restore inference to ONNX (default=enabled)\n"
                      " -w <digit>    CodeFormer Fidelity (Only ONNX, default=0.7)\n"
                      " -u            Face upsample (after face restore)\n"
                      " -z <string>   FaceUpsample model (ESRGAN)\n"
                      " -p            Use face parsing for accurate face masking (default=false)\n"
                      " -o <string>   Override image input path\n"
                      " -l <string>   Face detector model (default=y7, options: y7, y5, (RetinaFace: rt, mnet))\n"
                      " -h            Colorize grayscale photo with DeOldify Artistic\n"
                      " -n            No upsample\n"
                      " -a            Wait (pause execution)\n"
                      " -v            Verbose mode (detailed logging)";

    fprintf(stderr, str);
};

#if defined(AS_DLL)
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call,
                      LPVOID lpReserved)// reserved
{
    // Perform actions based on the reason for calling.
    switch (ul_reason_for_call) {
        case DLL_PROCESS_ATTACH:
            break;

        case DLL_THREAD_ATTACH: {
            DisableThreadLibraryCalls(hModule);
        } break;
        case DLL_THREAD_DETACH: {
        } break;
        case DLL_PROCESS_DETACH:
            break;
    }
    return TRUE;// Successful DLL_PROCESS_ATTACH.
}
#else

__declspec(noinline) char *getfilea(char *t) {
    char *str = 0;
    if (!t)
        return 0;
    if (str = strrchr(t, '/'))
        return str + 1;
    if (str = strrchr(t, '\\'))
        return str + 1;
    if (str == 0)
        return t;
    return str;
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
    std::wstring esr_model = L"./models/ESRGAN/4xNomos8kSC";
    char esr_modela[_MAX_PATH] = "./models/ESRGAN/4xNomos8kSC";
    std::wstring face_restore_model_onnx = L"./models/face_restore/codeformer_0_1_0";
    char gfp_modela[_MAX_PATH] = "./models/face_restore/codeformer_0_1_0";
    char color_modela[_MAX_PATH] = "./models/COLOR/deoldify_artistic";
    std::wstring color_model = L"./models/COLOR/deoldify_artistic";
    std::wstring out_path;
    char out_patha[_MAX_PATH];
    std::wstring fc_up_m;
    char cdf_upa[_MAX_PATH];
    std::wstring fc_det = L"y7";
    char fc_deta[_MAX_PATH] = "y7";

    //default processing params
    bool upsample = true;
    int model_scale = 0;
    int custom_scale = 0;
    int tilesize = 0;
    bool restore_face = false;
    bool verbose = false;
    bool wait = false;
    bool use_codeformer = false;
    bool use_infer_onnx = true;
    float codeformer_fidelity = 0.5;
    bool fc_up_ = false;
    bool useParse = false;
    bool color = false;
    float prob_face_thd = 0.5f;
    float nms_face_thd = 0.65f;

#ifdef _WIN32
    setlocale(LC_ALL, "");
#endif

    wchar_t opt;
    while ((opt = parsecmd(argc, argv, L"i:s:t:j:f:m:g:v:n:c:x:w:d:u:z:o:p:a:l:h")) != (wchar_t) 0) {
        switch (opt) {
            case L'i': {
                if (optarg) {
                    imagepath = optarg;
                    wcstombs(imagepatha, imagepath.data(), _MAX_PATH);
                }
            } break;
            case L's': {
                if (optarg)
                    model_scale = _wtoi(optarg);
            } break;
            case L't': {
                if (optarg)
                    tilesize = _wtoi(optarg);
            } break;
            case L'f': {
                restore_face = true;
            } break;
            case L'm': {
                if (optarg) {
                    esr_model = optarg;
                    wcstombs(esr_modela, esr_model.data(), _MAX_PATH);
                }
            } break;
            case L'g': {
                if (optarg) {
                    face_restore_model_onnx = optarg;
                    wcstombs(gfp_modela, face_restore_model_onnx.data(), _MAX_PATH);
                }
            } break;
            case L'h': {
                color = true;
            } break;
            case L'v': {
                verbose = true;
            } break;
            case L'c': {
                use_codeformer = true;
            } break;
            case L'd': {
                use_infer_onnx = false;
            } break;
            case L'w': {
                if (optarg)
                    codeformer_fidelity = _wtof(optarg);
            } break;
            case L'x': {
                if (optarg)
                    prob_face_thd = _wtof(optarg);
            } break;
            case L'n': {
                upsample = false;
            } break;
            case L'u': {
                fc_up_ = true;
            } break;
            case L'a': {
                wait = true;
            } break;
            case L'l': {
                if (optarg) {
                    fc_det = optarg;
                    wcstombs(fc_deta, fc_det.data(), _MAX_PATH);
                }
            } break;
            case L'p': {
                useParse = true;
            } break;
            case L'z': {
                if (optarg) {
                    fc_up_m = optarg;
                    wcstombs(cdf_upa, fc_up_m.data(), _MAX_PATH);
                }
            } break;
            case L'j': {
                if (optarg)
                    custom_scale = _wtoi(optarg);
            } break;
            case L'o': {
                if (optarg) {
                    out_path = optarg;
                    wcstombs(out_patha, out_path.data(), _MAX_PATH);
                }
            } break;
            case L'?': {
                print_usage();
                return 0;
            } break;
        }
    }

    if (imagepath.empty() || (false == upsample && false == restore_face)) {
        print_usage();
        if (wait)
            getch();
        return 0;
    }

    unsigned char *pixeldata = 0;
    int w{}, h{}, c{};

    cv::Mat image = cv::imread(imagepatha, cv::ImreadModes::IMREAD_COLOR_BGR);

    PipelineConfig_t pipeline_config_t;
    PipeLine pipe;

    pipeline_config_t.model_path = L"./models/";
    if (true == use_infer_onnx)
        pipeline_config_t.onnx = true;

    if (true == upsample) {
        pipeline_config_t.bg_upsample = upsample;
        pipeline_config_t.esr_model = esr_model;

        if (tilesize)
            pipeline_config_t.tilesize = tilesize;
        pipeline_config_t.model_scale = (model_scale == 0) ? pipe.getModelScale(esr_model) : model_scale;
    }

    if (custom_scale)
        pipeline_config_t.custom_scale = custom_scale;

    if (true == restore_face) {
        pipeline_config_t.face_upsample = fc_up_;
        pipeline_config_t.prob_thr = prob_face_thd;
        pipeline_config_t.face_restore = restore_face;
        pipeline_config_t.face_det_model = fc_det;
        pipeline_config_t.useParse = useParse;
        pipeline_config_t.w = codeformer_fidelity;
        pipeline_config_t.fc_up_model = fc_up_m;
        if (false == use_infer_onnx) {
            if (true == use_codeformer)
                pipeline_config_t.codeformer = true;
            else
                pipeline_config_t.codeformer = false;
        } else {
            pipeline_config_t.face_model = face_restore_model_onnx;
        }
    }

    if (true == color) {
        pipeline_config_t.Colorize = color;
        pipeline_config_t.colorize_m = color_model;
    }

    pipe.CreatePipeLine(pipeline_config_t);

    if (true == verbose) {
        fprintf(stderr, "Input image dimensions w: %d, h: %d, c: %d...\n", image.cols, image.rows, image.channels());
        if (custom_scale)
            fprintf(stderr, "Output image dimensions w: %d, h: %d, c: %d...\n", image.cols * custom_scale, image.rows * custom_scale, image.channels());
        else
            fprintf(stderr, "Output image dimensions w: %d, h: %d, c: %d...\n", image.cols * model_scale, image.rows * model_scale, c);

        fprintf(stderr, "tilesize: %d, ncnn_inf: %d, onnx_inf: %d, restore_face: %d,"
                        " model_scale: %d, upsample: %d, use_codeformer: %d\n"
                        " gfp_model_path: %s\n"
                        " esr_model_path: %s\n"
                        " heap_vram_budget: %d\n"
                        " colorize: %d\n"
                        " custom_scale: %d\n"
                        " codeformer face upsample: %d\n"
                        " codeformer fidelity: %.2f\n"
                        " face detect threshold: %.2f\n"
                        " face detect model: %s\n"
                        " OpenCV have OpenCL: %d\n"
                        " OpenCV uses OpenCL: %d\n"
                        " OpenCV cpu core uses: %d\n",
                tilesize ? tilesize : pipe.getEffectiveTilesize(), use_infer_onnx ? 0 : 1, use_infer_onnx, restore_face, (model_scale == 0) ? pipe.getModelScale(esr_model) : model_scale,
                upsample, use_codeformer, gfp_modela, esr_modela, ncnn::get_gpu_device(ncnn::get_default_gpu_index())->get_heap_budget(),
                color, custom_scale, fc_up_, codeformer_fidelity, prob_face_thd, fc_deta, cv::ocl::haveOpenCL(), cv::ocl::useOpenCL(), cv::getNumThreads());
    }

    cv::Mat result = pipe.Apply(image);

    //------------------------------------- save result image -------------------------------------
    {
        std::stringstream file;
        if (!out_path.empty()) {
            file << out_patha << std::ends;
        } else {
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
            if (color)
                file << "_color";

            file << ".png" << std::ends;
        }

        cv::imwrite(file.view().data(), result);
    }
    //------------------------------------- Save result image -------------------------------------


    fprintf(stderr, "Finish enjoy...\n");
    if (wait)
        getch();

    return 0;
};
#endif