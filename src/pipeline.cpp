// Image restore pipeline

#include "include/pipeline.h"
#include <numeric>

#if defined(_WIN32)
#if defined(USE_DM)
#include "include\helpers.h"
#endif
#endif

PipeLine::PipeLine(){};
PipeLine::~PipeLine() {
    Clear();
};

void PipeLine::Clear() {
    if (face_detector)
        delete face_detector;
    if (sessionOptions)
        delete sessionOptions;
    if (ortSession)
        delete ortSession;
    if (env)
        delete env;
    if (color)
        delete color;
    if (bg_upsampler)
        delete bg_upsampler;
    if (face_upsampler)
        delete face_upsampler;
    if (gfpgan_ncnn)
        delete gfpgan_ncnn;
    if (codeformer_ncnn)
        delete codeformer_ncnn;
    if (parsing_net)
        delete parsing_net;
    crops.clear();
    ncnn::destroy_gpu_instance();
};

int PipeLine::getModelScale(std::wstring str_bins) {
    //Heuristic model scale detection method
    if (str_bins.find(L"1x", 0) != std::string::npos || str_bins.find(L"x1", 0) != std::string::npos)
        return 1;
    else if (str_bins.find(L"2x", 0) != std::string::npos || str_bins.find(L"x2", 0) != std::string::npos)
        return 2;
    else if (str_bins.find(L"3x", 0) != std::string::npos || str_bins.find(L"x3", 0) != std::string::npos)
        return 3;
    else if (str_bins.find(L"4x", 0) != std::string::npos || str_bins.find(L"x4", 0) != std::string::npos)
        return 4;
    else if (str_bins.find(L"5x", 0) != std::string::npos || str_bins.find(L"x5", 0) != std::string::npos)
        return 5;
    else if (str_bins.find(L"8x", 0) != std::string::npos || str_bins.find(L"x8", 0) != std::string::npos)
        return 8;
    else if (str_bins.find(L"16x", 0) != std::string::npos || str_bins.find(L"x16", 0) != std::string::npos)
        return 16;

    return 0;
};

int PipeLine::getEffectiveTilesize() {

    uint32_t heap_budget = ncnn::get_gpu_device(ncnn::get_default_gpu_index())->get_heap_budget();//VRAM size
    int tilesize = 20;

    //calculate tilesize for VRAM consumption
    if (heap_budget > 1900)
        tilesize = 200;
    else if (heap_budget > 550)
        tilesize = 100;
    else if (heap_budget > 190)
        tilesize = 64;
    else
        tilesize = 32;
    return tilesize;
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

void PipeLine::paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample) {

    //Add padding for more accuracy determine face location

    if (pipe.custom_scale) {
        trans_matrix_inv *= pipe.custom_scale;
        trans_matrix_inv.at<double>(0, 2) += (double) 0.5 * (pipe.custom_scale - 1);
        trans_matrix_inv.at<double>(1, 2) += (double) 0.5 * (pipe.custom_scale - 1);
    } else {
        if (pipe.bg_upsample) {
            if (false == pipe.twox_mode) {
                trans_matrix_inv *= pipe.model_scale;
                trans_matrix_inv.at<double>(0, 2) += (double) 0.5 * (pipe.model_scale - 1);
                trans_matrix_inv.at<double>(1, 2) += (double) 0.5 * (pipe.model_scale - 1);
            } else {
                trans_matrix_inv *= pipe.model_scale * pipe.model_scale;
                trans_matrix_inv.at<double>(0, 2) += (double) 0.5 * ((pipe.model_scale * pipe.model_scale) - 1);
                trans_matrix_inv.at<double>(1, 2) += (double) 0.5 * ((pipe.model_scale * pipe.model_scale) - 1);
            }
        }
    }

    cv::Mat upscaled_face;

    double ups_f = 0.0;
    if (pipe.face_upsample) {
        ups_f = face_upsampler->scale;
        ncnn::Mat bg_presample(restored_face.cols, restored_face.rows, (void *) restored_face.data,
                               (size_t) restored_face.channels(), restored_face.channels());
        ncnn::Mat bg_upsamplencnn(restored_face.cols * ups_f, restored_face.rows * ups_f,
                                  (size_t) restored_face.channels(), restored_face.channels());

        fprintf(stderr, "Upsample face start...\n");

        face_upsampler->process(bg_presample, bg_upsamplencnn);
        cv::Mat dummy(bg_upsamplencnn.h, bg_upsamplencnn.w,
                      CV_8UC(restored_face.channels()), (void *) bg_upsamplencnn.data);
        upscaled_face = dummy.clone();

        trans_matrix_inv /= ups_f;
        trans_matrix_inv.at<double>(0, 2) *= ups_f;
        trans_matrix_inv.at<double>(1, 2) *= ups_f;
        trans_matrix_inv.at<double>(0, 2) -= 0.5 / (ups_f + 1) + 0.1;
        trans_matrix_inv.at<double>(1, 2) -= 0.5 / (ups_f + 1) + 0.1;

        fprintf(stderr, "Upsample face finish...\n");
    }

    cv::Mat inv_restored;
    if (pipe.face_upsample)
        cv::warpAffine(upscaled_face, inv_restored, trans_matrix_inv, bg_upsample.size(),
                       cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_CONSTANT);
    else
        cv::warpAffine(restored_face, inv_restored, trans_matrix_inv, bg_upsample.size(),
                       cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_CONSTANT);

    cv::Size ms_size;

    if (pipe.face_upsample)
        ms_size = cv::Size(512 * ups_f, 512 * ups_f);
    else
        ms_size = cv::Size(512, 512);

    cv::Mat mask;
    if (pipe.useParse) {
        const int num_class = 15;

        ncnn::Extractor ex = parsing_net->create_extractor();
        const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
        const float norm_vals[3] = {0.017058f, 0.017439f, 0.017361f};

        //const float mean_vals[3] = {0.485, 0.456, 0.406};
        //const float norm_vals[3] = {0.229, 0.224, 0.225};

        ncnn::Mat ncnn_in = ncnn::Mat::from_pixels(restored_face.data, ncnn::Mat::PIXEL_BGR2RGB, restored_face.cols, restored_face.rows);

        ncnn_in.substract_mean_normalize(mean_vals, norm_vals);
        fprintf(stderr, "Face parsing start...\n");
        ex.input("input", ncnn_in);
        ncnn::Mat output;
        ex.extract("output", output);

        //cv::Mat kx(output.h,output.w, CV_32FC1, output.data);
        //kx = (kx + 1.0f) / 2.0f * 255.0f; [-1, 1] -> [0, 255]
        //kx.convertTo(mask, CV_8UC1);

        mask = cv::Mat::zeros(cv::Size(512, 512), CV_8UC1);
        float *output_data = (float *) output.data;

        int out_h = mask.rows;
        int out_w = mask.cols;

#pragma omp parallel for
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                int maxk = 0;
                float tmp = output_data[0 * out_w * out_h + i * out_w + j];
                for (int k = 0; k < num_class; k++) {
                    if (tmp < output_data[k * out_w * out_h + i * out_w + j]) {
                        tmp = output_data[k * out_w * out_h + i * out_w + j];
                        maxk = 255;
                    }
                }
                mask.at<uchar>(i, j) = maxk;
            }
        }

        cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50, 50)));
        cv::GaussianBlur(mask, mask, cv::Size(101, 101), 11);
        cv::GaussianBlur(mask, mask, cv::Size(101, 101), 11);
        if (pipe.face_upsample)
            cv::resize(mask, mask, ms_size, 0, 0, cv::InterpolationFlags::INTER_LINEAR);
        fprintf(stderr, "Face parsing finished...\n");
    } else
        mask = cv::Mat::ones(ms_size, CV_8UC1) * 255;

    cv::UMat inv_mask;
    cv::warpAffine(mask.getUMat(cv::ACCESS_RW), inv_mask, trans_matrix_inv, bg_upsample.size(),
                   cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_CONSTANT);

    cv::Size krn_size;
    if (pipe.face_upsample)
        krn_size = cv::Size(4 * ups_f, 4 * ups_f);
    else
        krn_size = cv::Size(4, 4);

    cv::UMat inv_mask_erosion;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, krn_size);
    cv::erode(inv_mask, inv_mask_erosion, kernel);
    cv::Mat pasted_face;
    cv::bitwise_and(inv_restored, inv_restored, pasted_face, inv_mask_erosion);

    int total_face_area = cv::countNonZero(inv_mask_erosion);
    int w_edge = int(std::sqrt(total_face_area) / 20);
    int erosion_radius = w_edge * 2;
    cv::UMat inv_mask_center;
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erosion_radius, erosion_radius));
    cv::erode(inv_mask_erosion, inv_mask_center, kernel);

    int blur_size = w_edge * 2;
    cv::UMat inv_soft_mask_u;

    cv::GaussianBlur(inv_mask_center, inv_soft_mask_u, cv::Size(blur_size + 1, blur_size + 1),
                     0, 0, cv::BorderTypes::BORDER_DEFAULT);

    cv::Mat inv_soft_mask_f;
    inv_soft_mask_u.convertTo(inv_soft_mask_f, CV_32F, 1 / 255.f, 0.f);

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
};

int PipeLine::LaunchEngine(PipelineConfig_t &pipeline_config) {
    pipe = pipeline_config;

    Clear();

    {//Setup onnx inference
#if defined(_WIN32)
#if defined(USE_CD)
        sessionOptions = new Ort::SessionOptions();
        sessionOptions->EnableMemPattern();
        sessionOptions->SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        if (pipe.gpu) {
            Ort::Status status(OrtSessionOptionsAppendExecutionProvider_Tensorrt(*sessionOptions, 0));

            if (false == status.IsOK()) {
                OrtSessionOptionsAppendExecutionProvider_CUDA(*sessionOptions, 0);
            }
        }
#endif
#if defined(USE_DM)

        sessionOptions = new Ort::SessionOptions();
        sessionOptions->DisableMemPattern();
        sessionOptions->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        // By passing in an explicitly created DML device & queue, the DML execution provider sends work
        // to the desired device. If not used, the DML execution provider will create its own device & queue.
        if (pipe.gpu) {
            dml = CreateDmlDeviceAndCommandQueue("");
            ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void **>(&ortDmlApi));
            ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(
                    *sessionOptions,
                    get<0>(dml).Get(),
                    get<1>(dml).Get());
        }
#endif
#else if defined(__linux__)
        sessionOptions = new Ort::SessionOptions();
        sessionOptions->EnableMemPattern();
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

        if (pipe.gpu) {
            OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);//NVIDIA
        }
#endif

        env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "UPS_ONNX");
        // Load ONNX model into a session.
    }

    cv::setNumThreads(cv::getNumberOfCPUs());
    cv::ocl::haveOpenCL();
    cv::setUseOptimized(true);
    cv::ocl::setUseOpenCL(pipe.gpu);
    ncnn::create_gpu_instance();

    if (0 == pipe.tilesize)
        pipe.tilesize = getEffectiveTilesize();

    return 0;
};

cv::Mat image2tensor(const cv::Mat &inputImage) {

    cv::Mat blob = cv::dnn::blobFromImage(inputImage, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
    blob = blob * 2.0 - 1.0;

    return blob;
};

cv::Mat tensor2image(const float *outputData, const std::vector<int64_t> &outputShape) {
    int N = static_cast<int>(outputShape[0]);
    int C = static_cast<int>(outputShape[1]);
    int H = static_cast<int>(outputShape[2]);
    int W = static_cast<int>(outputShape[3]);

    size_t channelSize = static_cast<size_t>(H * W);

    std::vector<cv::Mat> channels;
    for (int i = 0; i < C; i++) {
        cv::Mat channel(H, W, CV_32F, const_cast<float *>(outputData + i * channelSize));
        channels.push_back(channel.clone());
    }

    cv::Mat image;
    cv::merge(channels, image);

    image = (image + 1.0f) / 2.0f * 255.0f;
    image.convertTo(image, CV_8UC(C));

    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

    return image;
};

cv::Mat PipeLine::inferFaceModel(
        const cv::Mat &input_img) {

    Ort::AllocatorWithDefaultOptions ortAllocator;

    auto inputName = ortSession->GetInputNameAllocated(0, ortAllocator);
    auto inputTypeInfo = ortSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto inputShape = inputTensorInfo.GetShape();

    auto outputName = ortSession->GetOutputNameAllocated(0, ortAllocator);
    auto outputTypeInfo = ortSession->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    auto outputShape = outputTensorInfo.GetShape();

    cv::Mat imgp = image2tensor(input_img);
    float inputTensorSize = 1;
    for (auto dim: inputShape) {
        inputTensorSize *= static_cast<float>(dim);
    }
    // For simplicity, this sample binds input/output buffers in system memory instead of DirectX resources.
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    //Fix for restoreformer
    if (pipe.face_model.find(L"restoreformer", 0) != std::string::npos) {
        if (0 > inputShape[0])
            inputShape[0] = 1;//Override batch size

        if (0 > outputShape[0])
            outputShape[0] = 1;
        if (0 > outputShape[2])
            outputShape[2] = 512;
        if (0 > outputShape[3])
            outputShape[3] = 512;
    }

    auto imageTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            (float *) imgp.data,
            inputTensorSize,
            inputShape.data(),
            inputShape.size());

    auto bindings = Ort::IoBinding::IoBinding(*ortSession);

    if (pipe.face_model.find(L"codeformer", 0) != std::string::npos) {
        std::array<int64_t, 1> fidelityShape = {1};
        std::vector<double> fidelityValue{pipe.w};
        auto fidelityTensor = Ort::Value::CreateTensor(
                memoryInfo,
                fidelityValue.data(),
                fidelityValue.size() * sizeof(double),
                fidelityShape.data(),
                fidelityShape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
        auto fidelityName = ortSession->GetInputNameAllocated(1, ortAllocator);
        bindings.BindInput(fidelityName.get(), fidelityTensor);
    }

    bindings.BindInput(inputName.get(), imageTensor);
    bindings.BindOutput(outputName.get(), memoryInfo);

    // Run the session to get inference results.
    Ort::RunOptions runOpts;

    fwprintf(stderr, L"Processing %s model...\n", pipe.face_model.c_str());

    ortSession->Run(runOpts, bindings);
    fprintf(stderr, "Processing onnx model finished...\n");
    bindings.SynchronizeOutputs();

    return tensor2image((float *) bindings.GetOutputValues()[0].GetTensorRawData(), outputShape);
};

cv::Mat PipeLine::Apply(const cv::Mat &input_img) {
    crops.clear();
    cv::Mat img_input = input_img.clone();
    cv::Mat img_upsample;
    cv::UMat img_upsample_umat;
    cv::Mat img_alpha;

    if (img_input.channels() == 1) {
        cv::cvtColor(img_input, img_input, cv::COLOR_GRAY2BGR);
    } else if (img_input.channels() == 4) {
        cv::extractChannel(img_input, img_alpha, 3);
        cv::cvtColor(img_input, img_input, cv::COLOR_BGRA2BGR);
    }

    if (1 == pipe.Colorize)
        color->process(img_input, img_input);

    if (pipe.bg_upsample) {
        int w{img_input.cols * pipe.model_scale}, h{img_input.rows * pipe.model_scale};

        ncnn::Mat bg_upsample(w, h,
                              (size_t) img_input.channels(), img_input.channels());

        ncnn::Mat bg_input(img_input.cols, img_input.rows, (void *) img_input.data,
                           (size_t) img_input.channels(), img_input.channels());

        fwprintf(stderr, L"Upscale image...\n");
        bg_upsampler->process(bg_input, bg_upsample);
        fwprintf(stderr, L"Upscale image finished...\n");

        if (false == pipe.twox_mode) {
            cv::Mat dummy(h, w,
                          CV_8UC3, (void *) bg_upsample.data);
            img_upsample = dummy.clone();
        } else {
            int w2{img_input.cols * (pipe.model_scale * pipe.model_scale)}, h2{img_input.rows * (pipe.model_scale * pipe.model_scale)};
            ncnn::Mat bg_input2(w, h, (void *) bg_upsample.data,
                                (size_t) img_input.channels(), img_input.channels());
            ncnn::Mat bg_upsample2(w2, h2,
                                   (size_t) img_input.channels(), img_input.channels());

            fwprintf(stderr, L"Upscale image 2x...\n");
            bg_upsampler->process(bg_input2, bg_upsample2);
            fwprintf(stderr, L"Upscale image 2x finished...\n");
            cv::Mat dummy(h2, w2,
                          CV_8UC3, (void *) bg_upsample2.data);
            img_upsample = dummy.clone();
        }

        if (pipe.custom_scale) {
            img_upsample_umat = img_upsample.getUMat(cv::ACCESS_RW);
            cv::resize(img_upsample_umat, img_upsample_umat, cv::Size(img_input.cols * pipe.custom_scale, img_input.rows * pipe.custom_scale), 0, 0, cv::InterpolationFlags::INTER_LANCZOS4);
            img_upsample = img_upsample_umat.getMat(cv::ACCESS_RW).clone();
        }
    } else {
        img_upsample = img_input.clone();
        if (pipe.custom_scale) {
            img_upsample_umat = img_upsample.getUMat(cv::ACCESS_RW);
            cv::resize(img_upsample_umat, img_upsample_umat, cv::Size(img_input.cols * pipe.custom_scale, img_input.rows * pipe.custom_scale), 0, 0, cv::InterpolationFlags::INTER_LANCZOS4);
            img_upsample = img_upsample_umat.getMat(cv::ACCESS_RW).clone();
        }
    }

    if (pipe.face_restore) {

        PipeResult_t pipe_result;
        fwprintf(stderr, L"Detecting faces...\n");
        face_detector->Process(img_input, (void *) &pipe_result);
        fwprintf(stderr, L"Detected %d faces\n", pipe_result.face_count);

        for (int i = 0; i != pipe_result.face_count; ++i) {
            fwprintf(stderr, L"Processing %d face...\n", i + 1);
            crops.push_back(pipe_result.object[i].trans_img.clone());
            if (pipe.onnx) {
                cv::Mat restored_face = inferFaceModel(
                        pipe_result.object[i].trans_img);
                crops.push_back(restored_face.clone());
                fwprintf(stderr, L"Paste %d face in photo...\n", i + 1);
                paste_faces_to_input_image(restored_face, pipe_result.object[i].trans_inv, img_upsample);
            } else {
                if (pipe.codeformer) {
                    CodeFormerResult_t res;
                    pipe_result.codeformer_result.push_back(res);
                    codeformer_ncnn->Process(pipe_result.object[i].trans_img, pipe_result.codeformer_result[i]);
                    crops.push_back(pipe_result.codeformer_result[i].restored_face.clone());
                    fwprintf(stderr, L"Paste %d face in photo...\n", i + 1);
                    paste_faces_to_input_image(pipe_result.codeformer_result[i].restored_face, pipe_result.object[i].trans_inv, img_upsample);

                } else {
                    ncnn::Mat gfpgan_result;
                    gfpgan_ncnn->process(pipe_result.object[i].trans_img, gfpgan_result);

                    cv::Mat restored_face;
                    to_ocv(gfpgan_result, restored_face);
                    crops.push_back(restored_face.clone());
                    paste_faces_to_input_image(restored_face, pipe_result.object[i].trans_inv, img_upsample);
                }
            }
        }
    }

    if (2 == pipe.Colorize)
        color->process(img_upsample, img_upsample);

    if (!img_alpha.empty()) {
        cv::resize(img_alpha, img_alpha, cv::Size(img_upsample.cols, img_upsample.rows), 0, 0, cv::InterpolationFlags::INTER_CUBIC);
        cv::cvtColor(img_upsample, img_upsample, cv::COLOR_BGR2BGRA);
        cv::insertChannel(img_alpha, img_upsample, 3);
    }

    return img_upsample.clone();
};

PipeLine *PipeLine::getApi() {
    return new PipeLine();
};

int PipeLine::load_bg_esr_model(PipelineConfig_t &cfg) {

    bool succ = false;
    if (bg_upsampler) {
        delete bg_upsampler;
        bg_upsampler = nullptr;
    }
    if (cfg.esr_model.empty()) {
        pipe.bg_upsample = false;
        pipe.model_scale = 0;
        return 1;
    }

    pipe.esr_model = cfg.esr_model;
    pipe.model_scale = cfg.model_scale;
    pipe.tta_mode = cfg.tta_mode;

    if (0 != cfg.tilesize)
        pipe.tilesize = cfg.tilesize;
    else
        pipe.tilesize = getEffectiveTilesize();

    std::wstringstream str_param;
    str_param << pipe.esr_model << ".param" << std::ends;
    std::wstringstream str_bin;
    str_bin << pipe.esr_model << ".bin" << std::ends;

    if (pipe.model_scale == 0) {
        int scale = getModelScale(str_bin.str());

        if (scale) {
            bg_upsampler = new RealESRGAN(pipe.gpu, pipe.tta_mode);
            bg_upsampler->scale = scale;
            pipe.bg_upsample = true;
            succ = true;
        } else {
            pipe.bg_upsample = false;
            fwprintf(stderr, L"Error autodetect scale of this face upscale model please add x[Scale] or [Scale]x to filename of model\n"
                             "bg upscale disabled...");
            return 0;
        }
    } else {
        bg_upsampler = new RealESRGAN(pipe.gpu, pipe.tta_mode);
        bg_upsampler->scale = pipe.model_scale;
        pipe.bg_upsample = true;
        succ = true;
    }

    if (succ)
        if (bg_upsampler->scale) {
            bg_upsampler->prepadding = 10;
            pipe.model_scale = bg_upsampler->scale;

            bg_upsampler->tilesize = pipe.tilesize;

            fwprintf(stderr, L"Loading background upsample model...\n");
            if (bg_upsampler->load(str_param.view().data(), str_bin.view().data()) == 0) {
                fwprintf(stderr, L"Loading background upsample finished...\n");
            } else {
                delete bg_upsampler;
                bg_upsampler = nullptr;
                pipe.bg_upsample = false;
                return 0;
            }
        }
};

int PipeLine::load_face_model(PipelineConfig_t &cfg) {
    if (cfg.face_model.empty()) {
        if (ortSession) {
            delete ortSession;
            ortSession = nullptr;
        }
        pipe.face_restore = false;
        return 1;
    }

    pipe.onnx = true;
    pipe.face_restore = true;
    pipe.face_model = cfg.face_model;

    std::wstring path;
    path = pipe.face_model;
    path += L".onnx";

    if (ortSession) {
        delete ortSession;
        ortSession = nullptr;
    }
    fwprintf(stderr, L"Loading onnx model...\n");
    ortSession = new Ort::Session(*env, path.c_str(), *sessionOptions);
    fwprintf(stderr, L"Loading onnx model finished...\n");
};

int PipeLine::load_face_up_model(PipelineConfig_t &cfg) {
    if (cfg.fc_up_model.empty()) {
        if (face_upsampler) {
            delete face_upsampler;
            face_upsampler = nullptr;
            pipe.fc_up_model = L"";
        }
        pipe.face_upsample = false;
        return 1;
    }
    pipe.face_upsample = true;
    pipe.fc_up_model = cfg.fc_up_model;

    if (0 != cfg.tilesize)
        pipe.tilesize = cfg.tilesize;
    else
        pipe.tilesize = getEffectiveTilesize();

    if (face_upsampler) {
        delete face_upsampler;
        face_upsampler = nullptr;
    }

    std::wstringstream str_param;
    str_param << pipe.fc_up_model << ".param" << std::ends;
    std::wstringstream str_bin;
    str_bin << pipe.fc_up_model << ".bin" << std::ends;

    int scale = getModelScale(str_bin.str());

    if (scale) {
        face_upsampler = new RealESRGAN(pipe.gpu);
        face_upsampler->scale = scale;
        face_upsampler->prepadding = 10;
        face_upsampler->tilesize = pipe.tilesize;

        fwprintf(stderr, L"Loading face upsample model...\n");
        if (face_upsampler->load(str_param.view().data(), str_bin.view().data()) == 0) {
            fwprintf(stderr, L"Loading face upsample finished...\n");
        } else {
            pipe.face_upsample = false;
            delete face_upsampler;
            face_upsampler = nullptr;
            return 0;
        }
    } else {
        pipe.face_upsample = false;
        fwprintf(stderr, L"Error autodetect scale of this face upscale model please add x[Scale] or [Scale]x to filename of model\n"
                         "Face upscale disabled...");
        return 0;
    }

    return 1;
};

int PipeLine::load_face_det_model(PipelineConfig_t &cfg) {
    if (cfg.face_det_model.empty())
        return 0;

    pipe.face_det_model = cfg.face_det_model;

    if (face_detector) {
        delete face_detector;
        face_detector = nullptr;
    }
    if (pipe.face_det_model.find(L"y7", 0) != std::string::npos)
        face_detector = new Faceyolov7_lite_e();
    if (pipe.face_det_model.find(L"y5", 0) != std::string::npos)
        face_detector = new Face_yolov5_bl();
    if (pipe.face_det_model.find(L"rt", 0) != std::string::npos)
        face_detector = new FaceR(pipe.gpu);
    if (pipe.face_det_model.find(L"mnet", 0) != std::string::npos)
        face_detector = new FaceR(pipe.gpu, true);

    int ret = face_detector->Load(pipe.model_path);
    if (ret == 0) {
        face_detector->setThreshold(pipe.prob_thr, pipe.nms_thr);
    } else {
        pipe.face_restore = false;
        delete face_detector;
        face_detector = nullptr;
        return 0;
    }
    return 1;
};

int PipeLine::changeScaleFactor(PipelineConfig_t &cfg) {
    pipe.custom_scale = cfg.custom_scale;
    return 1;
};

int PipeLine::changeCodeformerFiledily(PipelineConfig_t &cfg) {
    pipe.w = cfg.w;
    return 1;
};

int PipeLine::setFaceDetectorThreshold(PipelineConfig_t &cfg) {
    if (face_detector) {
        pipe.prob_thr = cfg.prob_thr;
        pipe.nms_thr = cfg.nms_thr;
        face_detector->setThreshold(pipe.prob_thr, pipe.nms_thr);

        return 1;
    }
    return 0;
};

int PipeLine::setUseParse(PipelineConfig_t &cfg) {
    if (false == cfg.useParse) {
        pipe.useParse = false;
        if (parsing_net) {
            parsing_net->clear();
            delete parsing_net;
            parsing_net = nullptr;
        }
    } else {
        pipe.useParse = true;
        parsing_net = new ncnn::Net();
        parsing_net->opt.num_threads = ncnn::get_cpu_count();
        parsing_net->opt.use_vulkan_compute = pipe.gpu;
        std::wstring model_param = pipe.model_path + L"/face_pars/face_parsing.param";
        std::wstring model_bin = pipe.model_path + L"/face_pars/face_parsing.bin";

        fwprintf(stderr, L"Loading face parsing model from %s...\n", model_bin.c_str());

        int ret_param{}, ret_bin{};

        FILE *f = _wfopen(model_param.c_str(), L"rb");
        if (f) {
            ret_param = parsing_net->load_param(f);
            fclose(f);
            f = 0;
        }
        f = _wfopen(model_bin.c_str(), L"rb");
        if (f) {
            ret_bin = parsing_net->load_model(f);
            fclose(f);
        }

        if (!ret_param || !ret_bin) {
            fwprintf(stderr, L"Loading face parsing model finished...\n");
        } else {
            pipe.useParse = false;
            delete parsing_net;
            parsing_net = nullptr;
            return 0;
        }
    }

    return 1;
};

int PipeLine::switchToNCNNFaceModels(PipelineConfig_t &cfg) {
    pipe.onnx = cfg.onnx;
    pipe.codeformer = cfg.codeformer;

    if (false == pipe.onnx) {
        if (pipe.codeformer) {
            if (codeformer_ncnn) {
                delete codeformer_ncnn;
                codeformer_ncnn = nullptr;
            }
            if (gfpgan_ncnn) {
                delete gfpgan_ncnn;
                gfpgan_ncnn = nullptr;
            }

            codeformer_ncnn = new CodeFormer(pipe.gpu);

            fprintf(stderr, "Loading codeformer model...\n");
            int ret = codeformer_ncnn->Load(pipe.model_path);
            if (ret == 0) {
                fprintf(stderr, "Loading codeformer finished...\n");
            } else {
                delete codeformer_ncnn;
                codeformer_ncnn = nullptr;
                pipe.face_restore = false;
                return 0;
            }
        } else {
            if (gfpgan_ncnn) {
                delete gfpgan_ncnn;
                gfpgan_ncnn = nullptr;
            }
            if (codeformer_ncnn) {
                delete codeformer_ncnn;
                codeformer_ncnn = nullptr;
            }

            gfpgan_ncnn = new GFPGAN();

            fprintf(stderr, "Loading GFPGANCleanv1-NoCE-C2 model from /models/GFPGANCleanv1-NoCE-C2-*...\n");
            if (gfpgan_ncnn->load(pipe.model_path) == 0) {
                fprintf(stderr, "Loading GFPGANCleanv1-NoCE-C2 model finished...\n");
            } else {
                pipe.face_restore = false;
                delete gfpgan_ncnn;
                gfpgan_ncnn = nullptr;
                return 0;
            }
        }
    } else {
        if (gfpgan_ncnn) {
            delete gfpgan_ncnn;
            gfpgan_ncnn = nullptr;
        }
        if (codeformer_ncnn) {
            delete codeformer_ncnn;
            codeformer_ncnn = nullptr;
        }
    }

    return 1;
};

int PipeLine::load_color_model(PipelineConfig_t &cfg) {
    pipe.Colorize = cfg.Colorize;
    pipe.colorize_m = cfg.colorize_m;

    if (0 == pipe.Colorize) {
        if (color) {
            delete color;
            color = nullptr;
        }
    } else {
        if (color) {
            delete color;
            color = nullptr;
        }
        color = new ColorSiggraph(pipe.gpu);

        fprintf(stderr, "Loading colorization model...\n");
        if (color->load(*sessionOptions, pipe.colorize_m) == 0) {
            fprintf(stderr, "Loading colorization model finished...\n");
            return 1;
        } else {
            pipe.Colorize = 0;
            delete color;
            color = nullptr;
            return 0;
        }
    }
};

int PipeLine::changeColorState(PipelineConfig_t &cfg) {
    if (pipe.colorize_m.empty())
        pipe.Colorize = 0;
    else
        pipe.Colorize = cfg.Colorize;

    return 1;
};

int PipeLine::setESRTTAand2x(PipelineConfig_t &cfg) {
    if (true == pipe.bg_upsample) {
        pipe.tta_mode = cfg.tta_mode;
        pipe.twox_mode = cfg.twox_mode;

        bg_upsampler->enableTTA(pipe.tta_mode);
    }

    return 1;
};

std::vector<cv::Mat> &PipeLine::getCrops() {
    return crops;
};