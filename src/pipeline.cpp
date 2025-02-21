// Face restore pipeline

#include "include/pipeline.h"
#include <numeric>

#if defined(_WIN32)
extern unsigned char *wic_decode_image(const wchar_t *filepath, int *w, int *h, int *c);
extern int wic_encode_image(const wchar_t *filepath, int w, int h, int c, void *bgrdata);
#include "include\helpers.h"
#endif

PipeLine::PipeLine() {
}
PipeLine::~PipeLine() {
    Clear();
}

void PipeLine::Clear() {
    if (face_detector)
        delete face_detector;
    if (env)
        delete env;
    if (ortSession)
        delete ortSession;
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
}

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
}

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
        trans_matrix_inv.at<double>(0, 2) += (double) pipe.custom_scale / 2;
        trans_matrix_inv.at<double>(1, 2) += (double) pipe.custom_scale / 2;
    } else {
        trans_matrix_inv.at<double>(0, 2) += (double) pipe.model_scale / 2;
        trans_matrix_inv.at<double>(1, 2) += (double) pipe.model_scale / 2;
    }

    cv::Mat upscaled_face;

    double ups_f = 0.0;
    if (pipe.face_upsample) {
        ups_f = face_up_NCNN_->scale;
        ncnn::Mat bg_presample(restored_face.cols, restored_face.rows, (void *) restored_face.data,
                               (size_t) restored_face.channels(), restored_face.channels());
        ncnn::Mat bg_upsamplencnn(restored_face.cols * ups_f, restored_face.rows * ups_f,
                                  (size_t) restored_face.channels(), restored_face.channels());

        fprintf(stderr, "Upsample face start...\n");

        face_up_NCNN_->process(bg_presample, bg_upsamplencnn);
        cv::Mat dummy(bg_upsamplencnn.h, bg_upsamplencnn.w,
                      (restored_face.channels() == 3) ? CV_8UC3 : CV_8UC4, (void *) bg_upsamplencnn.data);
        upscaled_face = dummy.clone();

        //
        trans_matrix_inv /= ups_f;
        trans_matrix_inv.at<double>(0, 2) *= ups_f;
        trans_matrix_inv.at<double>(1, 2) *= ups_f;
        trans_matrix_inv.at<double>(0, 2) -= ups_f / 2;
        trans_matrix_inv.at<double>(1, 2) -= ups_f / 2;

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
        const int target_width = 512;
        const int target_height = 512;
        const int num_class = 15;

        ncnn::Extractor ex = parsing_net.create_extractor();
        const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
        const float norm_vals[3] = {0.017058f, 0.017439f, 0.017361f};
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
}

int PipeLine::CreatePipeLine(PipelineConfig_t &pipeline_config) {
    pipe = pipeline_config;

    Clear();

    {//Setup onnx inference
#if defined(_WIN32)
        sessionOptions.DisableMemPattern();
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        // By passing in an explicitly created DML device & queue, the DML execution provider sends work
        // to the desired device. If not used, the DML execution provider will create its own device & queue.
        if (pipe.gpu) {
            dml = CreateDmlDeviceAndCommandQueue("");
            Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void **>(&ortDmlApi)));
            Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(
                    sessionOptions,
                    get<0>(dml).Get(),
                    get<1>(dml).Get()));
        }
#else if defined(__linux__)
        if (pipe.gpu) {

            sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
#if defined(ROCmbuild)
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ROCM(sessionOptions, 0));//AMD
#endif
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));//NVIDIA
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

    if (true == pipe.bg_upsample) {
        std::wstringstream str_param;
        str_param << pipe.esr_model << ".param" << std::ends;
        std::wstringstream str_bin;
        str_bin << pipe.esr_model << ".bin" << std::ends;

        if (pipe.model_scale == 0) {
            int scale = getModelScale(str_bin.str());
            if (scale) {
                pipe.model_scale = scale;
            } else {
                pipe.bg_upsample = false;
                fprintf(stderr, "Error autodetect scale of this face upscale model please add x[Scale] or [Scale]x to filename of model\n"
                                "bg upscale disabled...");
            }
        }

        if (pipe.model_scale) {
            bg_upsample_md = new RealESRGAN(pipe.gpu);
            bg_upsample_md->scale = pipe.model_scale;
            bg_upsample_md->prepadding = 10;

            bg_upsample_md->tilesize = getEffectiveTilesize();

            fprintf(stderr, "Loading background upsample model...\n");
            bg_upsample_md->load(str_param.view().data(), str_bin.view().data());
            fprintf(stderr, "Loading background upsample finished...\n");
        }
    }

    if (true == pipe.onnx) {
        if (!pipe.face_model.empty()) {
            fprintf(stderr, "Loading onnx model...\n");
            ortSession = new Ort::Session(*env, pipe.face_model.c_str(), sessionOptions);
            fprintf(stderr, "Loading onnx model finished...\n");
        }
    } else {
        if (pipe.codeformer) {
            codeformer_NCNN_ = new CodeFormer(pipe.gpu);
            fprintf(stderr, "Loading codeformer model...\n");
            int ret = codeformer_NCNN_->Load(pipe.model_path);
            if (ret < 0) {
                return -1;
            }
            fprintf(stderr, "Loading codeformer finished...\n");
        } else {
            gfpgan_NCNN_ = new GFPGAN();
            fprintf(stderr, "Loading GFPGANCleanv1-NoCE-C2 model from /models/GFPGANCleanv1-NoCE-C2-*...\n");
            gfpgan_NCNN_->load(pipe.model_path);
            fprintf(stderr, "Loading GFPGANCleanv1-NoCE-C2 model finished...\n");
        }
    }

    if (!pipe.face_det_model.empty()) {
        if (pipe.face_det_model.find(L"y7", 0) != std::string::npos)
            face_detector = new Faceyolov7_lite_e();
        if (pipe.face_det_model.find(L"y5", 0) != std::string::npos)
            face_detector = new Face_yolov5_bl();
        if (pipe.face_det_model.find(L"rt", 0) != std::string::npos)
            face_detector = new FaceR(pipe.gpu);

        int ret = face_detector->Load(pipe.model_path);
        if (ret < 0) {
            return -1;
        }
        if (pipe.custom_scale)
            face_detector->setScale(pipe.custom_scale);
        else
            face_detector->setScale(pipe.model_scale);

        face_detector->setThreshold(pipe.prob_thr, pipe.nms_thr);
    }

    if (pipe.useParse) {
        parsing_net.opt.num_threads = ncnn::get_cpu_count();
        parsing_net.opt.use_vulkan_compute = pipe.gpu;
        std::wstring model_param = pipe.model_path + L"/face_pars/face_parsing.param";
        std::wstring model_bin = pipe.model_path + L"/face_pars/face_parsing.bin";

        fwprintf(stderr, L"Loading face parsing model from %s...\n", model_bin.c_str());

        FILE *f = _wfopen(model_param.c_str(), L"rb");
        int ret_param = parsing_net.load_param(f);
        fclose(f);

        f = _wfopen(model_bin.c_str(), L"rb");
        int ret_bin = parsing_net.load_model(f);
        fclose(f);
        fwprintf(stderr, L"Loading face parsing model finished...\n");
    }

    if (pipe.face_upsample) {
        std::wstringstream str_param;
        str_param << pipe.fc_up_model << ".param" << std::ends;
        std::wstringstream str_bin;
        str_bin << pipe.fc_up_model << ".bin" << std::ends;

        int scale = getModelScale(str_bin.str());
        if (scale) {
            face_up_NCNN_ = new RealESRGAN(pipe.gpu);
            face_up_NCNN_->scale = scale;
        } else {
            pipe.face_upsample = false;
            fprintf(stderr, "Error autodetect scale of this face upscale model please add x[Scale] or [Scale]x to filename of model\n"
                            "Face upscale disabled...");
        }

        if (face_up_NCNN_->scale) {
            face_up_NCNN_->prepadding = 10;
            face_up_NCNN_->tilesize = getEffectiveTilesize();

            fprintf(stderr, "Loading face upsample model...\n");
            face_up_NCNN_->load(str_param.view().data(), str_bin.view().data());
            fprintf(stderr, "Loading face upsample finished...\n");
        }
    }

    if (pipe.colorize) {
        color = new ColorSiggraph(pipe.gpu);

        fprintf(stderr, "Loading colorization model...\n");
        color->load(pipe.model_path.c_str());
        fprintf(stderr, "Loading colorization model finished...\n");
    }

    return 0;
}

cv::Mat preprocessImage(const cv::Mat &inputImage) {
    cv::Mat img;

    // 1. Приведение к типу float32 (если изображение загружено, например, через imread, то оно имеет тип CV_8UC3)
    inputImage.convertTo(img, CV_32F);

    //// 2. (Опционально) Изменяем размер до 512x512, если требуется
    // cv::resize(img, img, cv::Size(512, 512), 0, 0, cv::INTER_LINEAR);

    //// 3. Масштабирование в диапазон [0,1]
    img = img / 255.0;

    //// 4. Конвертация из BGR в RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    //// 5. Создание blob: cv::dnn::blobFromImage выполняет автоматическую перестановку из HWC в CHW
    ////    Здесь scaleFactor равен 1.0, поскольку мы уже делим на 255.
    ////    Параметр swapRB установлен в false, так как мы уже перевели в RGB.
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
    // Теперь blob имеет форму [1, 3, H, W] (NCHW)

    //// 6. Нормализация: операция (img - 0.5)/0.5 равносильна (2*img - 1)
    ////    Применяем нормализацию по всему blob
    blob = blob * 2.0 - 1.0;

    //// 7. Blob уже имеет размерность батча (N=1), поэтому дополнительных действий не требуется.
    return blob;
}

// Функция postProcessImage преобразует выходной тензор модели в cv::Mat.
cv::Mat postProcessImage(const float *outputData, const std::vector<int64_t> &outputShape) {
    // Ожидаем, что выход имеет форму [N, C, H, W]
    if (outputShape.size() != 4) {
        fprintf(stderr, "Expected output tensor shape with 4 dimensions (N, C, H, W).");
    }

    int N = static_cast<int>(outputShape[0]);
    int C = static_cast<int>(outputShape[1]);
    int H = static_cast<int>(outputShape[2]);
    int W = static_cast<int>(outputShape[3]);

    if (N != 1) {
        fprintf(stderr, "postProcessImage supports only batch size of 1.");
    }
    if (C != 3) {
        fprintf(stderr, "postProcessImage supports only 3-channel output.");
    }

    // Размер каждого канала
    size_t channelSize = static_cast<size_t>(H * W);

    // Извлекаем данные по каналам и создаём для каждого канал cv::Mat.
    // Здесь предполагается, что данные расположены подряд: сначала весь первый канал, затем второй, затем третий.
    std::vector<cv::Mat> channels;
    for (int i = 0; i < C; i++) {
        // Создаем заголовок для матрицы, без копирования данных (для безопасности делаем clone ниже)
        cv::Mat channel(H, W, CV_32F, const_cast<float *>(outputData + i * channelSize));
        channels.push_back(channel.clone());// clone гарантирует непрерывность данных
    }

    // Объединяем каналы в одно изображение формата HWC (RGB)
    cv::Mat image;
    cv::merge(channels, image);

    // Преобразуем значения из диапазона [-1, 1] в [0, 255].
    // Формула: image_out = (image + 1) / 2 * 255
    image = (image + 1.0f) / 2.0f * 255.0f;
    image.convertTo(image, CV_8UC3);

    // Если требуется, преобразуем изображение из RGB в BGR для корректного отображения/сохранения OpenCV
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

    return image;
}

cv::Mat PipeLine::inferONNXModel(
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

    cv::Mat imgp = preprocessImage(input_img);
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
        // ===== Создаём второй входной тензор (fidelity) =====
        // Форма для fidelity – одномерный тензор с одним элементом.
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

    return postProcessImage((float *) bindings.GetOutputValues()[0].GetTensorRawData(), outputShape);
}

cv::Mat PipeLine::Apply(const cv::Mat &input_img) {
    cv::Mat imgd = input_img.clone();
    cv::Mat bg_upsample_ocv;
    cv::UMat bg_upsample_ocv_u;

    if (pipe.bg_upsample) {
        ncnn::Mat bg_upsample_ncnn(imgd.cols * pipe.model_scale, imgd.rows * pipe.model_scale,
                                   (size_t) imgd.channels(), imgd.channels());

        ncnn::Mat bg_presample_ncnn(imgd.cols, imgd.rows, (void *) imgd.data,
                                    (size_t) imgd.channels(), imgd.channels());

        fwprintf(stderr, L"Upscale image...\n");
        bg_upsample_md->process(bg_presample_ncnn, bg_upsample_ncnn);
        fwprintf(stderr, L"Upscale image finished...\n");

        cv::Mat dummy(imgd.rows * pipe.model_scale, imgd.cols * pipe.model_scale,
                      (imgd.channels() == 3) ? CV_8UC3 : CV_8UC4, (void *) bg_upsample_ncnn.data);
        bg_upsample_ocv = dummy.clone();

        if (pipe.custom_scale) {
            bg_upsample_ocv_u = bg_upsample_ocv.getUMat(cv::ACCESS_RW);
            cv::resize(bg_upsample_ocv_u, bg_upsample_ocv_u, cv::Size(imgd.cols * pipe.custom_scale, imgd.rows * pipe.custom_scale), 0, 0, cv::InterpolationFlags::INTER_LINEAR);
            bg_upsample_ocv = bg_upsample_ocv_u.getMat(cv::ACCESS_RW).clone();
        }
    } else {
        bg_upsample_ocv = imgd.clone();
        if (pipe.custom_scale) {
            bg_upsample_ocv_u = bg_upsample_ocv.getUMat(cv::ACCESS_RW);
            cv::resize(bg_upsample_ocv_u, bg_upsample_ocv_u, cv::Size(imgd.cols * pipe.custom_scale, imgd.rows * pipe.custom_scale), 0, 0, cv::InterpolationFlags::INTER_LINEAR);
            bg_upsample_ocv = bg_upsample_ocv_u.getMat(cv::ACCESS_RW).clone();
        }
    }

    if (pipe.face_restore) {

        PipeResult_t pipe_result;
        fwprintf(stderr, L"Detecting faces...\n");
        face_detector->Process(input_img, (void *) &pipe_result);
        fwprintf(stderr, L"Detected %d faces\n", pipe_result.face_count);
        crops.clear();
        for (int i = 0; i != pipe_result.face_count; ++i) {
            fwprintf(stderr, L"%s process %d face...\n", getfilew((wchar_t *) pipe.face_model.c_str()), i + 1);
            crops.push_back(pipe_result.object[i].trans_img);
            if (pipe.onnx) {
                cv::Mat restored_face = inferONNXModel(
                        pipe_result.object[i].trans_img);
                crops.push_back(restored_face);
                fwprintf(stderr, L"Paste %d face in photo...\n", i + 1);
                paste_faces_to_input_image(restored_face, pipe_result.object[i].trans_inv, bg_upsample_ocv);
            } else {
                if (pipe.codeformer) {
                    CodeFormerResult_t res;
                    pipe_result.codeformer_result.push_back(res);
                    codeformer_NCNN_->Process(pipe_result.object[i].trans_img, pipe_result.codeformer_result[i]);
                    crops.push_back(pipe_result.codeformer_result[i].restored_face);
                    fwprintf(stderr, L"Paste %d face in photo...\n", i + 1);
                    paste_faces_to_input_image(pipe_result.codeformer_result[i].restored_face, pipe_result.object[i].trans_inv, bg_upsample_ocv);

                } else {
                    ncnn::Mat gfpgan_result;
                    gfpgan_NCNN_->process(pipe_result.object[i].trans_img, gfpgan_result);

                    cv::Mat restored_face;
                    to_ocv(gfpgan_result, restored_face);
                    crops.push_back(restored_face);
                    paste_faces_to_input_image(restored_face, pipe_result.object[i].trans_inv, bg_upsample_ocv);
                }
            }
        }
    }

    if (pipe.colorize)
        color->process(bg_upsample_ocv, bg_upsample_ocv);

    return bg_upsample_ocv.clone();
}

PipeLine *PipeLine::getApi() {
    return new PipeLine();
}

void PipeLine::changeSettings(int type, PipelineConfig_t &cfg) {

    switch (type) {//Change ESR
        case AI_SettingsOp::CHANGE_ESR: {
            bool succ = false;
            if (bg_upsample_md) {
                delete bg_upsample_md;
                bg_upsample_md = nullptr;
            }
            if (cfg.esr_model.empty()) {
                pipe.bg_upsample = false;
                pipe.model_scale = 0;
                if (0 == pipe.custom_scale)
                    face_detector->setScale(pipe.model_scale);
                else
                    face_detector->setScale(cfg.custom_scale);
                return;
            }

            pipe.esr_model = cfg.esr_model;
            pipe.model_scale = cfg.model_scale;

            std::wstringstream str_param;
            str_param << pipe.esr_model << ".param" << std::ends;
            std::wstringstream str_bin;
            str_bin << pipe.esr_model << ".bin" << std::ends;

            if (pipe.model_scale == 0) {
                int scale = getModelScale(str_bin.str());

                if (scale) {
                    bg_upsample_md = new RealESRGAN(pipe.gpu);
                    bg_upsample_md->scale = scale;
                    pipe.bg_upsample = true;
                    succ = true;
                } else {
                    pipe.bg_upsample = false;
                    fwprintf(stderr, L"Error autodetect scale of this face upscale model please add x[Scale] or [Scale]x to filename of model\n"
                                     "bg upscale disabled...");
                }
            } else {
                bg_upsample_md = new RealESRGAN(pipe.gpu);
                bg_upsample_md->scale = pipe.model_scale;
                pipe.bg_upsample = true;
                succ = true;
            }

            if (succ)
                if (bg_upsample_md->scale) {
                    bg_upsample_md->prepadding = 10;
                    pipe.model_scale = bg_upsample_md->scale;

                    bg_upsample_md->tilesize = getEffectiveTilesize();

                    if (0 == pipe.custom_scale)
                        face_detector->setScale(pipe.model_scale);
                    else
                        face_detector->setScale(cfg.custom_scale);

                    fwprintf(stderr, L"Loading background upsample model...\n");
                    bg_upsample_md->load(str_param.view().data(), str_bin.view().data());
                    fwprintf(stderr, L"Loading background upsample finished...\n");
                }
        } break;
        case AI_SettingsOp::CHANGE_GFP: {//Change GFP

            if (cfg.face_model.empty()) {
                if (ortSession) {
                    delete ortSession;
                    ortSession = nullptr;
                }
                pipe.face_restore = false;
                return;
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
            ortSession = new Ort::Session(*env, path.c_str(), sessionOptions);
            fwprintf(stderr, L"Loading onnx model finished...\n");

        } break;
        case AI_SettingsOp::CHANGE_FACE_DET: {//Change Face detector

            if (cfg.face_det_model.empty())
                return;

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

            int ret = face_detector->Load(pipe.model_path);
            if (pipe.custom_scale)
                face_detector->setScale(pipe.custom_scale);
            else
                face_detector->setScale(pipe.model_scale);

            face_detector->setThreshold(pipe.prob_thr, pipe.nms_thr);
        } break;
        case AI_SettingsOp::CHANGE_FACE_UP_M: {//Change face upsample
            if (cfg.fc_up_model.empty()) {
                if (face_up_NCNN_) {
                    delete face_up_NCNN_;
                    face_up_NCNN_ = nullptr;
                    pipe.fc_up_model = L"";
                }
                pipe.face_upsample = false;
                return;
            }
            pipe.face_upsample = true;
            pipe.fc_up_model = cfg.fc_up_model;

            if (face_up_NCNN_) {
                delete face_up_NCNN_;
                face_up_NCNN_ = nullptr;
            }

            std::wstringstream str_param;
            str_param << pipe.fc_up_model << ".param" << std::ends;
            std::wstringstream str_bin;
            str_bin << pipe.fc_up_model << ".bin" << std::ends;

            int scale = getModelScale(str_bin.str());

            if (scale) {
                face_up_NCNN_ = new RealESRGAN(pipe.gpu);
                face_up_NCNN_->scale = scale;
                face_up_NCNN_->prepadding = 10;
                face_up_NCNN_->tilesize = getEffectiveTilesize();

                fwprintf(stderr, L"Loading face upsample model...\n");
                face_up_NCNN_->load(str_param.view().data(), str_bin.view().data());
                fwprintf(stderr, L"Loading face upsample finished...\n");
            } else {
                pipe.face_upsample = false;
                fwprintf(stderr, L"Error autodetect scale of this face upscale model please add x[Scale] or [Scale]x to filename of model\n"
                                 "Face upscale disabled...");
            }

        } break;
        case AI_SettingsOp::CHANGE_SCALE_FACTOR: {//Override scale factor
            pipe.custom_scale = cfg.custom_scale;
            if (0 == cfg.custom_scale)
                face_detector->setScale(pipe.model_scale);
            else
                face_detector->setScale(cfg.custom_scale);
        } break;
        case AI_SettingsOp::CHANGE_CODEFORMER_FID: {//Change codeformer fidelity
            pipe.w = cfg.w;
        } break;
        case AI_SettingsOp::CHANGE_FACEDECT_THD: {//Change facedect threshold
            pipe.prob_thr = cfg.prob_thr;
            pipe.nms_thr = cfg.nms_thr;
            face_detector->setThreshold(pipe.prob_thr, pipe.nms_thr);
        } break;
        case AI_SettingsOp::CHANGE_FACE_PARSE: {

            if (false == cfg.useParse) {
                pipe.useParse = false;
                parsing_net.clear();
            } else {
                pipe.useParse = true;

                parsing_net.opt.num_threads = ncnn::get_cpu_count();
                parsing_net.opt.use_vulkan_compute = pipe.gpu;
                std::wstring model_param = pipe.model_path + L"/face_pars/face_parsing.param";
                std::wstring model_bin = pipe.model_path + L"/face_pars/face_parsing.bin";

                fwprintf(stderr, L"Loading face parsing model from %s...\n", model_bin.c_str());

                FILE *f = _wfopen(model_param.c_str(), L"rb");
                int ret_param = parsing_net.load_param(f);
                fclose(f);

                f = _wfopen(model_bin.c_str(), L"rb");
                int ret_bin = parsing_net.load_model(f);
                fclose(f);
                fwprintf(stderr, L"Loading face parsing model finished...\n");
            }
        } break;
        case AI_SettingsOp::CHANGE_INFER: {
            pipe.onnx = cfg.onnx;
            pipe.codeformer = cfg.codeformer;

            if (false == pipe.onnx) {
                if (pipe.codeformer) {
                    if (codeformer_NCNN_) {
                        delete codeformer_NCNN_;
                        codeformer_NCNN_ = nullptr;
                    }
                    if (gfpgan_NCNN_) {
                        delete gfpgan_NCNN_;
                        gfpgan_NCNN_ = nullptr;
                    }

                    codeformer_NCNN_ = new CodeFormer(pipe.gpu);

                    fprintf(stderr, "Loading codeformer model...\n");
                    int ret = codeformer_NCNN_->Load(pipe.model_path);
                    fprintf(stderr, "Loading codeformer finished...\n");
                } else {
                    if (gfpgan_NCNN_) {
                        delete gfpgan_NCNN_;
                        gfpgan_NCNN_ = nullptr;
                    }
                    if (codeformer_NCNN_) {
                        delete codeformer_NCNN_;
                        codeformer_NCNN_ = nullptr;
                    }

                    gfpgan_NCNN_ = new GFPGAN();

                    fprintf(stderr, "Loading GFPGANCleanv1-NoCE-C2 model from /models/GFPGANCleanv1-NoCE-C2-*...\n");
                    gfpgan_NCNN_->load(pipe.model_path);
                    fprintf(stderr, "Loading GFPGANCleanv1-NoCE-C2 model finished...\n");
                }
            } else {
                if (gfpgan_NCNN_) {
                    delete gfpgan_NCNN_;
                    gfpgan_NCNN_ = nullptr;
                }
                if (codeformer_NCNN_) {
                    delete codeformer_NCNN_;
                    codeformer_NCNN_ = nullptr;
                }
            }
        } break;
        case AI_SettingsOp::CHANGE_COLOR: {
            pipe.colorize = cfg.colorize;

            if (false == pipe.colorize) {
                if (color) {
                    delete color;
                    color = nullptr;
                }
            } else {
                color = new ColorSiggraph(pipe.gpu);

                fprintf(stderr, "Loading colorization model...\n");
                color->load(pipe.model_path.c_str());
                fprintf(stderr, "Loading colorization model finished...\n");
            }
        } break;
    }
};

std::vector<cv::Mat> &PipeLine::getCrops() {
    return crops;
}
