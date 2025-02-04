#include "face.h"
#include "gfpgan.h"
#include "pipeline.h"
#include "realesrgan.h"
#include <cstdio>
#include <iostream>
#include <net.h>
#include <string>
#include <string_view>

#include "wic_image.h"

#define VER "1.00"

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
    fprintf(stderr, " -s <digit> model scale factor (default=4)\n");
    fprintf(stderr, " -j <digit> custom output scale factor\n");
    fprintf(stderr, " -t <digit> tile size (default=auto)\n");
    fprintf(stderr, " -f restore faces (GFPGAN 1.4) (default=0)\n");
    fprintf(stderr, " -m <string> esrgan model name (default=./models/x4nomos8ksc)\n");
    fprintf(stderr, " -g <string> gfpgan model path (default=./models/gfpgan_1.4)\n");
    fprintf(stderr, " -x <digit> YOLOV face detection threshold (default=0,5) (0.3..0.7 recommended)\n");
    fprintf(stderr, " -c use CodeFormer face restore model\n");
    fprintf(stderr, " -d swith CodeFormer infer to onnx\n");
    fprintf(stderr, " -w CodeFormer Fidelity (Only onnx) (default=0.7)\n");
    fprintf(stderr, " -u CodeFormer FaceUpsample\n");
    fprintf(stderr, " -z CodeFormer FaceUpsample model\n");
    fprintf(stderr, " -p use gfpgan-ncnn infer instead of onnx(DirectML prefer) (only GFPGANCleanv1-NoCE-C2 model and CPU backend)\n");
    fprintf(stderr, " -n no upsample\n");
    fprintf(stderr, " -v verbose\n");
};

static void to_ocv(const ncnn::Mat &result, cv::Mat &out) {
    cv::Mat cv_result_32F = cv::Mat::zeros(cv::Size(512, 512), CV_32FC3);
    for (int i = 0; i < result.h; i++) {
        for (int j = 0; j < result.w; j++) {
            cv_result_32F.at<cv::Vec3f>(i, j)[2] = (result.channel(0)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[1] = (result.channel(1)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[0] = (result.channel(2)[i * result.w + j] + 1) / 2;
        }
    }

    cv::Mat cv_result_8U;
    cv_result_32F.convertTo(cv_result_8U, CV_8UC3, 255.0, 0);

    cv_result_8U.copyTo(out);
};

static void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample) {
    trans_matrix_inv.at<float>(0, 2) += 1.0;
    trans_matrix_inv.at<float>(1, 2) += 1.0;

    cv::Mat inv_restored;
    cv::warpAffine(restored_face, inv_restored, trans_matrix_inv, bg_upsample.size(), 1, 0);

    cv::Mat mask = cv::Mat::ones(cv::Size(512, 512), CV_8UC1) * 255;
    cv::Mat inv_mask;
    cv::warpAffine(mask, inv_mask, trans_matrix_inv, bg_upsample.size(), 1, 0);

    cv::Mat inv_mask_erosion;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4));
    cv::erode(inv_mask, inv_mask_erosion, kernel);
    cv::Mat pasted_face;
    cv::bitwise_and(inv_restored, inv_restored, pasted_face, inv_mask_erosion);

    int total_face_area = cv::countNonZero(inv_mask_erosion);
    int w_edge = int(std::sqrt(total_face_area) / 20);
    int erosion_radius = w_edge * 2;
    cv::Mat inv_mask_center;
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erosion_radius, erosion_radius));
    cv::erode(inv_mask_erosion, inv_mask_center, kernel);

    int blur_size = w_edge * 2;
    cv::Mat inv_soft_mask;
    cv::GaussianBlur(inv_mask_center, inv_soft_mask, cv::Size(blur_size + 1, blur_size + 1), 0, 0, 4);

    cv::Mat inv_soft_mask_f;
    inv_soft_mask.convertTo(inv_soft_mask_f, CV_32F, 1 / 255.f, 0.f);

#pragma omp parallel for
    for (int h = 0; h < bg_upsample.rows; ++h) {
        cv::Vec3b *img_ptr = bg_upsample.ptr<cv::Vec3b>(h);
        cv::Vec3b *face_ptr = pasted_face.ptr<cv::Vec3b>(h);
        float *mask_ptr = inv_soft_mask_f.ptr<float>(h);
        for (int w = 0; w < bg_upsample.cols; ++w) {
            img_ptr[w][0] = img_ptr[w][0] * (1 - mask_ptr[w]) + face_ptr[w][0] * mask_ptr[w];
            img_ptr[w][1] = img_ptr[w][1] * (1 - mask_ptr[w]) + face_ptr[w][1] * mask_ptr[w];
            img_ptr[w][2] = img_ptr[w][2] * (1 - mask_ptr[w]) + face_ptr[w][2] * mask_ptr[w];
        }
    }
}

bool pathisfolderw(wchar_t *c) {
    if (c) {
        if (wcsrchr(c, L'\\'))
            return true;
        if (wcsrchr(c, L'/'))
            return true;
        return false;
    }
};
bool pathisfoldera(char *c) {
    if (c) {
        if (strrchr(c, '/'))
            return true;
        if (strrchr(c, '\\'))
            return true;
        return false;
    }
};
wchar_t *getfilew(wchar_t *t) {
    wchar_t *str = 0;
    if (t) {
        if (str = wcsrchr(t, L'/'))
            return str + 1;
        if (str = wcsrchr(t, L'\\'))
            return str + 1;
    }
    return str;
};
char *getfilea(char *t) {
    char *str = 0;
    if (t) {
        if (str = strrchr(t, '/'))
            return str + 1;
        if (str = strrchr(t, '\\'))
            return str + 1;
    }
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
    std::wstring esr_model = L"./models/4xNomos8kSC";
    char esr_modela[_MAX_PATH] = "./models/4xNomos8kSC";
    std::wstring gfp_model = L"./models/gfpgan_1.4";
    char gfp_modela[_MAX_PATH] = "./models/gfpgan_1.4";
    std::wstring cdf_up;
    char cdf_upa[_MAX_PATH];

    //default processing params
    bool upsample = true;
    int model_scale = 0;
    int custom_scale = 0;
    int tilesize = 20;
    bool restore_face = false;
    bool verbose = false;
    bool ncnn_gfp = false;
    bool use_codeformer = false;
    bool use_codeformer_onnx = false;
    float codeformer_fidelity = 0.7;
    bool codeformer_fc_up = false;
    float prob_face_thd = 0.5f;
    float nms_face_thd = 0.65f;

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:s:t:j:f:p:m:g:v:n:h:c:x:w:d:u:z")) != (wchar_t) 0) {
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
            case L'p': {
                ncnn_gfp = true;
            } break;
            case L'c': {
                use_codeformer = true;
            } break;
            case L'd': {
                use_codeformer_onnx = true;
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
                codeformer_fc_up = true;
            } break;
            case L'z': {
                cdf_up = optarg;
                wcstombs(cdf_upa, cdf_up.data(), _MAX_PATH);
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

    if (false == upsample && false == restore_face) {
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
                fprintf(stderr, "Error autodetect scale of this upscale model please use -s flag to set model scale");
                ncnn::destroy_gpu_instance();
                return 0;
            }
        }

    if (true == verbose) {
        fprintf(stderr, "Input image dimensions w: %d, h: %d, c: %d...\n", w, h, c);
        fprintf(stderr, "tilesize: %d, ncnn_gfp: %d, onnx_cdf: %d, restore_face: %d, model_scale: %d, upsample: %d, use_codeformer: %d\n"
                        " gfp_model_path: %s\n"
                        " esr_model_path: %s\n"
                        " heap_vram_budget: %d\n"
                        " custom_scale: %d\n"
                        " codeformer face upsample: %d\n"
                        " codeformer fidelity: %.1f\n"
                        " face_detect_threshold: %.1f\n",
                tilesize, ncnn_gfp, use_codeformer_onnx, restore_face, model_scale, upsample, use_codeformer, gfp_modela, esr_modela, heap_budget, custom_scale, codeformer_fc_up, codeformer_fidelity, prob_face_thd);
    }
    ncnn::Mat bg_upsamplencnn(w * model_scale, h * model_scale, (size_t) c, c);
    ncnn::Mat bg_presample(w, h, (void *) pixeldata, (size_t) c, c);
    cv::Mat bg_upsamplecv;
    cv::Mat img_faces(h, w, (c == 3) ? CV_8UC3 : CV_8UC4, pixeldata);

    std::vector<cv::Mat> trans_img;
    std::vector<cv::Mat> trans_matrix_inv;
    std::vector<Object> objects;

    std::wstringstream str;
    char stra[_MAX_PATH];

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
        real_esrgan.process(bg_presample, bg_upsamplencnn);
        fprintf(stderr, "Upscale image finished...\n");

        str << imagepath << L"_" << getfilew(esr_model.data()) << L"_ms" << model_scale << L"_cs" << custom_scale << ".png" << std::ends;

        wic_encode_image(str.view().data(), bg_upsamplencnn.w, bg_upsamplencnn.h, bg_upsamplencnn.elempack, bg_upsamplencnn.data);
    } else {
        if (custom_scale)
            cv::resize(img_faces, bg_upsamplecv, cv::Size(img_faces.cols * custom_scale, img_faces.rows * custom_scale), 0, 0, cv::InterpolationFlags::INTER_CUBIC);
        else
            img_faces.copyTo(bg_upsamplecv);
        str << imagepath << L"_" << getfilew(gfp_model.data()) << L"_s" << model_scale << L"_cs" << custom_scale << "_interpolated"
            << ".png" << std::ends;
        wic_encode_image(str.view().data(), bg_upsamplecv.cols, bg_upsamplecv.rows, bg_upsamplecv.elemSize(), bg_upsamplecv.data);
    }
    wcstombs(stra, str.view().data(), _MAX_PATH);

    if (upsample) {
        if (custom_scale) {
            cv::Mat pre = cv::imread(stra, 1);
            cv::Mat up;
            cv::resize(pre, up, cv::Size(img_faces.cols * custom_scale, img_faces.rows * custom_scale), 0, 0, cv::InterpolationFlags::INTER_LANCZOS4);
            cv::imwrite(stra, up);
        }
    }

    if (true == restore_face) {
        char path[_MAX_PATH];
        wcstombs(path, str.view().data(), _MAX_PATH);
        cv::Mat img_faces_upsamle = cv::imread(path, 1);

        if (use_codeformer) {
            wsdsb::PipelineConfig_t pipeline_config_t;
            pipeline_config_t.model_path = "./models/";
            if (use_codeformer_onnx) {
                pipeline_config_t.onnx = true;
                pipeline_config_t.ncnn = false;
                pipeline_config_t.face_upsample = codeformer_fc_up;

                if (custom_scale)
                    pipeline_config_t.scale = custom_scale;
                else
                    pipeline_config_t.scale = model_scale;

                pipeline_config_t.w = codeformer_fidelity;
                strcpy_s(pipeline_config_t.name, 255, stra);
                pipeline_config_t.up_model = cdf_up;
            }

            wsdsb::PipeLine pipe;
            pipe.CreatePipeLine(pipeline_config_t);
            pipe.Apply(img_faces, img_faces_upsamle);
        } else {
            Face face_detector;
            fprintf(stderr, "Loading YOLOV5 face detector model from /models/yolov5-blazeface...\n");
            face_detector.load("./models/yolov5-blazeface.param", "./models/yolov5-blazeface.bin");
            fprintf(stderr, "Loading YOLOV5 face detector model finished...\n");

            fprintf(stderr, "Detecting faces...\n");
            face_detector.detect(img_faces, objects, prob_face_thd, nms_face_thd);
            fprintf(stderr, "Detected %d faces\n", objects.size());
            GFPGAN gfpgan;//GFPGANCleanv1-NoCE-C2

            if (true == ncnn_gfp) {
                fprintf(stderr, "Loading GFPGANv1 face detector model from /models/GFPGANCleanv1-NoCE-C2-*...\n");
                gfpgan.load("./models/GFPGANCleanv1-NoCE-C2-encoder.param", "./models/GFPGANCleanv1-NoCE-C2-encoder.bin", "./models/GFPGANCleanv1-NoCE-C2-style.bin");
                fprintf(stderr, "Loading GFPGAN model finished...\n");
            }

            if (custom_scale)
                face_detector.align_warp_face(img_faces_upsamle, objects, trans_matrix_inv, trans_img, custom_scale);

            face_detector.align_warp_face(img_faces_upsamle, objects, trans_matrix_inv, trans_img, model_scale);
            int n_f{};
            for (auto &x: trans_img) {
                if (false == ncnn_gfp) {
                    std::stringstream str;
                    str << imagepatha << "_" << n_f << "_crop.png" << std::ends;
                    cv::imwrite(str.view().data(), x);

                    fprintf(stderr, "Processing face %d...\n", n_f + 1);
                    std::stringstream str2;
                    str2 << "python gfpgan_onnx.py --model_path " << gfp_modela << ".onnx"
                         << " --image_path " << str.view() << std::ends;
                    system(str2.view().data());

                    cv::Mat restored_face = cv::imread("output.jpg", 1);
                    std::stringstream str3;
                    str3 << imagepatha << "_" << n_f << "_crop_" << getfilea(gfp_modela) << "_upsampled.png" << std::ends;
                    cv::imwrite(str3.view().data(), restored_face);
                    fprintf(stderr, "paste face %d into image...\n", n_f + 1);
                    paste_faces_to_input_image(restored_face, trans_matrix_inv[n_f], img_faces_upsamle);
                } else {
                    ncnn::Mat gfpgan_result;
                    fprintf(stderr, "Processing face %d...\n", n_f + 1);
                    gfpgan.process(trans_img[n_f], gfpgan_result);

                    cv::Mat restored_face;
                    to_ocv(gfpgan_result, restored_face);
                    fprintf(stderr, "paste face %d into image...\n", n_f + 1);
                    paste_faces_to_input_image(restored_face, trans_matrix_inv[n_f], img_faces_upsamle);
                }
                ++n_f;
            }
        }
        std::stringstream file;
        file << imagepatha;
        if (upsample)
            file << "_" << getfilea(esr_modela);
        if (restore_face) {
            if (use_codeformer) {
                file << "_codeformer";
                if (use_codeformer_onnx) {
                    file << "_w" << codeformer_fidelity;
                    if (codeformer_fc_up)
                        file << "_fu_" << getfilea(cdf_upa);
                }
            } else {
                file << "_" << getfilea(gfp_modela);
            }
        }

        file << "_ms" << model_scale;
        file << "_cs" << custom_scale;

        file << ".png" << std::ends;

        cv::imwrite(file.view().data(), img_faces_upsamle);
    }
#if _WIN32
    CoUninitialize();
#endif

    ncnn::destroy_gpu_instance();

    fprintf(stderr, "Finish enjoy...\n");

    return 0;
};