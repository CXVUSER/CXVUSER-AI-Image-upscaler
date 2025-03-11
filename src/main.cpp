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

#define VER "1.03"

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
    const char *str = R"(
CXVUSER AI MegaPixel XL Super-Black edition Upscale solution )" VER R"(, Welcome...
This project uses (onnx,ncnn inference) with Vulkan and DirectML compute...

Usage: this_binary [options]...

 -i <img> path to image
 -s <digit> model scale factor (default=autodetect)
 -j <digit> custom output scale factor
 -t <digit> tile size (default=auto)
 -f restore faces (default=codeformer)
 -m <string> esrgan model name (default=./models/ESRGAN/4xNomos8kSC)
 -g <string> gfpgan(or same as gfp) model path (default=./models/face_restore/codeformer_0_1_0.onnx)
 -x <digit> face detection threshold (default=0,5) (0,3..0,7 recommended)
 -c use CodeFormer face restore model (ncnn)
 -d swith face restore infer to onnx
 -w <digit> CodeFormer Fidelity (Only onnx) (default=0,7)
 -u Face Upsample (after face restore)
 -z <string> FaceUpsample model (ESRGAN)
 -p Use face parsing for accurate face masking (default=false)
 -o <string> override image input path
 -l <string> Face detector model (default=y7) (y7,y5,rt(retinaface R50))
 -h Colorize grayscale photo with Siggraph17
 -n no upsample
 -a wait
 -v verbose
)";

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
    std::wstring face_restore_model_onnx = L"./models/face_restore/codeformer_0_1_0.onnx";
    char gfp_modela[_MAX_PATH] = "./models/face_restore/codeformer_0_1_0.onnx";
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
    bool use_infer_onnx = false;
    float codeformer_fidelity = 0.5;
    bool fc_up_ = false;
    bool useParse = false;
    bool color = false;
    float prob_face_thd = 0.5f;
    float nms_face_thd = 0.65f;

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:s:t:j:f:m:g:v:n:c:x:w:d:u:z:o:p:a:l:h")) != (wchar_t) 0) {
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
                use_infer_onnx = true;
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
#endif

    if (imagepath.empty() || (false == upsample && false == restore_face)) {
        print_usage();
        if (wait)
            getch();
        return 0;
    }

    unsigned char *pixeldata = 0;
    int w{}, h{}, c{};

    int haveOpenCL{};
    int useOpenCL{};
    // if (haveOpenCL = cv::ocl::haveOpenCL())
    //    cv::ocl::setUseOpenCL(true);
    //useOpenCL = cv::ocl::useOpenCL();
    cv::setNumThreads(cv::getNumberOfCPUs());

    cv::Mat image = cv::imread(imagepatha, cv::ImreadModes::IMREAD_COLOR_BGR);

    PipelineConfig_t pipeline_config_t;
    PipeLine pipe;

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
                color, custom_scale, fc_up_, codeformer_fidelity, prob_face_thd, fc_deta, haveOpenCL, useOpenCL, cv::getNumThreads());
    }

    pipeline_config_t.model_path = L"./models/";
    if (use_infer_onnx)
        pipeline_config_t.onnx = true;

    pipeline_config_t.bg_upsample = upsample;
    pipeline_config_t.esr_model = esr_model;

    pipeline_config_t.face_upsample = fc_up_;
    pipeline_config_t.prob_thr = prob_face_thd;

    pipeline_config_t.custom_scale = custom_scale;
    pipeline_config_t.model_scale = (model_scale == 0) ? pipe.getModelScale(esr_model) : model_scale;

    pipeline_config_t.w = codeformer_fidelity;
    pipeline_config_t.fc_up_model = fc_up_m;

    pipeline_config_t.face_restore = restore_face;

    if (restore_face) {
        if (!use_infer_onnx) {
            if (use_codeformer)
                pipeline_config_t.codeformer = true;
            else
                pipeline_config_t.codeformer = false;
        } else {
            pipeline_config_t.face_model = face_restore_model_onnx;
        }
    }

    pipeline_config_t.face_det_model = fc_det;
    pipeline_config_t.useParse = useParse;
    pipeline_config_t.colorize = color;

    pipe.CreatePipeLine(pipeline_config_t);

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

    ncnn::destroy_gpu_instance();

    fprintf(stderr, "Finish enjoy...\n");
    if (wait)
        getch();

    return 0;
};
#endif