#include "include/ColorSiggraph.h"
#include "layer.h"

class Sig17Slice : public ncnn::Layer {
public:
    Sig17Slice() {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        //int outc = channels * 4;
        int outc = channels;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++) {
            const float *ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float *outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Sig17Slice)

ColorSiggraph::ColorSiggraph(bool gpu) {
    this->gpu = gpu;
}

ColorSiggraph::~ColorSiggraph() {
    if (env)
        delete env;
    if (ortSession)
        delete ortSession;
}

int ColorSiggraph::load(Ort::SessionOptions &sessOpt, std::wstring &model) {
    if (model.find(L"siggraph17", 0) != std::string::npos) {
        if (gpu)
            net.opt.use_vulkan_compute = true;
        else
            net.opt.use_vulkan_compute = false;

        net.register_custom_layer("Sig17Slice", Sig17Slice_layer_creator);

        std::wstring model_param = model;
        model_param += L".param";

        std::wstring model_bin = model;
        model_bin += L".bin";

        FILE *f = _wfopen(model_param.c_str(), L"rb");
        if (f) {
            int ret = net.load_param(f);
            fclose(f);
            if (ret < 0) {
                fwprintf(stderr, L"open param file %s failed\n", model_param.c_str());
                return -1;
            }
        } else {
            fwprintf(stderr, L"open param file %s failed\n", model_param.c_str());
            return -1;
        }

        f = _wfopen(model_bin.c_str(), L"rb");
        if (f) {
            int ret = net.load_model(f);
            fclose(f);
            if (ret < 0) {
                fwprintf(stderr, L"open bin file %s failed\n", model_bin.c_str());
                return -1;
            }
        } else {
            fwprintf(stderr, L"open bin file %s failed\n", model_bin.c_str());
            return -1;
        }
        type = 0;
    } else if (model.find(L"deoldify", 0) != std::string::npos) {
        std::wstring model_param = model;
        model_param += L".onnx";
        env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DeOldify");
        ortSession = new Ort::Session(*env, model_param.c_str(), sessOpt);
        type = model.find(L"artistic", 0) != std::string::npos ? 2 : 1;
    } else if (model.find(L"ddcolor", 0) != std::string::npos) {
        std::wstring model_param = model;
        model_param += L".onnx";
        env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DDColor");
        ortSession = new Ort::Session(*env, model_param.c_str(), sessOpt);
        type = model.find(L"tiny", 0) != std::string::npos ? 4 : 3;
    }

    return 0;
}

void ColorSiggraph::process_Siggraph17(const cv::Mat &inimage, cv::Mat &outimage) const {
    //fixed input size for the pretrained network
    const int W_in = 256;
    const int H_in = 256;
    cv::UMat lab, Base_img, L_u;
    cv::Mat input_img;
    Base_img = inimage.clone().getUMat(cv::ACCESS_RW);

    //normalize levels
    Base_img.convertTo(Base_img, CV_32F, 1.0 / 255.0);

    //Convert BGR to LAB color space format
    cvtColor(Base_img, lab, cv::COLOR_BGR2Lab);

    //Extract L channel
    cv::extractChannel(lab, L_u, 0);

    //Resize to input shape 256x256
    resize(L_u.getMat(cv::ACCESS_RW), input_img, cv::Size(W_in, H_in));

    //We subtract 50 from the L channel (for mean centering)
    //input_img -= 50;

    //convert to NCNN::MAT
    ncnn::Mat in_LAB_L(input_img.cols, input_img.rows, 1, (void *) input_img.data);
    in_LAB_L = in_LAB_L.clone();

    ncnn::Extractor ex = net.create_extractor();
    //set input, output lyers
    ex.input("input", in_LAB_L);

    //inference network
    ncnn::Mat out;
    //ex.extract("out_ab", out);
    ex.extract("out_ab", out);

    //create LAB material
    cv::Mat colored_LAB(out.h, out.w, CV_32FC2);
    //Extract ab channels from ncnn:Mat out
    memcpy((uchar *) colored_LAB.data, out.data, out.w * out.h * 2 * sizeof(float));

    //get separsted LAB channels a&b
    cv::Mat a(out.h, out.w, CV_32F, (float *) out.data);
    cv::Mat b(out.h, out.w, CV_32F, (float *) out.data + out.w * out.h);

    //Resize a, b channels to origina image size
    cv::resize(a, a, Base_img.size());
    cv::resize(b, b, Base_img.size());

    //merge channels, and convert back to BGR
    cv::Mat chn[] = {L_u.getMat(cv::ACCESS_RW), a, b};
    cv::merge(chn, 3, lab);
    cvtColor(lab, lab, cv::COLOR_Lab2BGR);
    //normalize values to 0->255
    lab.convertTo(lab, CV_8UC3, 255);
    lab.copyTo(outimage);
}

void ColorSiggraph::process_deoldify(const cv::Mat &inimage, cv::Mat &outimage) const {

    // Загрузка изображения
    cv::UMat image = inimage.clone().getUMat(cv::ACCESS_RW);

    // Конвертация в LAB и выделение канала L
    cv::UMat targetLAB, targetL, A, B;
    cv::cvtColor(image, targetLAB, cv::COLOR_BGR2Lab);
    std::vector<cv::UMat> lab_channels;
    split(targetLAB, lab_channels);
    targetL = lab_channels[0].clone();

    // Конвертация в градации серого и обратно в RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);

    int h = image.rows;
    int w = image.cols;

    // Изменение размера изображения
    int r_factor = (type == 2) ? 256 : 512;// Примерное значение
    cv::resize(image, image, cv::Size(r_factor, r_factor));

    image.convertTo(image, CV_32F);

    cv::Mat nchw_data = cv::dnn::blobFromImage(image, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);

    // Подготовка входных данных
    Ort::AllocatorWithDefaultOptions ortAllocator;
    auto inputName = ortSession->GetInputNameAllocated(0, ortAllocator);
    auto inputTypeInfo = ortSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto inputShape = inputTensorInfo.GetShape();

    inputShape[2] = r_factor;
    inputShape[3] = r_factor;

    auto outputName = ortSession->GetOutputNameAllocated(0, ortAllocator);
    auto outputTypeInfo = ortSession->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    auto outputShape = outputTensorInfo.GetShape();

    outputShape[2] = r_factor;
    outputShape[3] = r_factor;

    float inputTensorSize = 1;
    for (auto dim: inputShape) {
        inputTensorSize *= static_cast<float>(dim);
    }

    // Создание входного тензора
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                              (float *) nchw_data.data,
                                                              inputTensorSize,
                                                              inputShape.data(),
                                                              inputShape.size());

    auto bindings = Ort::IoBinding::IoBinding(*ortSession);

    bindings.BindInput(inputName.get(), input_tensor);
    bindings.BindOutput(outputName.get(), memoryInfo);

    Ort::RunOptions runOpts;
    ortSession->Run(runOpts, bindings);
    bindings.SynchronizeOutputs();

    // Получение результатов
    float *output_data = (float *) bindings.GetOutputValues()[0].GetTensorRawData();
    //cv::Mat data(256, 256, CV_32FC4, output_data);

    //// Объединяем в OpenCV-формат (HWC)
    //std::vector<cv::Mat> result_image;
    //cv::dnn::imagesFromBlob(data, result_image);
    // Размер каждого канала
    size_t channelSize = static_cast<size_t>(r_factor * r_factor);

    // Извлекаем данные по каналам и создаём для каждого канал cv::Mat.
    // Здесь предполагается, что данные расположены подряд: сначала весь первый канал, затем второй, затем третий.
    std::vector<cv::Mat> channels;
    for (int i = 0; i < outputShape[1]; i++) {
        // Создаем заголовок для матрицы, без копирования данных (для безопасности делаем clone ниже)
        cv::Mat channel(r_factor, r_factor, CV_32F, const_cast<float *>(output_data + i * channelSize));
        channels.push_back(channel.clone());// clone гарантирует непрерывность данных
    }

    // Объединяем каналы в одно изображение формата HWC (RGB)
    cv::UMat colorized;
    cv::merge(channels, colorized);
    
    colorized.convertTo(colorized, CV_8UC3);
    cv::cvtColor(colorized, colorized, cv::COLOR_BGR2RGB);

    // Постобработка изображения
    cv::resize(colorized, colorized, cv::Size(w, h));
    cv::GaussianBlur(colorized, colorized, cv::Size(13, 13), 0);

    cv::UMat colorzLab;
    cv::cvtColor(colorized, colorzLab, cv::COLOR_BGR2Lab);
    cv::split(colorzLab, lab_channels);
    A = lab_channels[1];
    B = lab_channels[2];

    cv::UMat a(w, h, CV_8UC1);
    cv::UMat b(w, h, CV_8UC1);

    cv::resize(lab_channels[1], a, cv::Size(w, h));
    cv::resize(lab_channels[2], b, cv::Size(w, h));

    // Объединение каналов L из исходного изображения и A, B из colorized
    cv::UMat result;
    std::vector<cv::UMat> merged_channels = {targetL, a, b};
    merge(merged_channels, result);

    cvtColor(result, result, cv::COLOR_Lab2BGR);
    result.copyTo(outimage);
}

void ColorSiggraph::process_DDColor(const cv::Mat &inimage, cv::Mat &outimage) const {
    // Загрузка изображения
    cv::UMat image = inimage.clone().getUMat(cv::ACCESS_RW);
    image.convertTo(image,CV_32F, 1.0 / 255.0);

    // Конвертация в LAB и выделение канала L
    cv::UMat targetLAB, targetL, A, B;
    cv::cvtColor(image, targetLAB, cv::COLOR_BGR2Lab);
    std::vector<cv::UMat> lab_channels;
    split(targetLAB, lab_channels);
    targetL = lab_channels[0].clone();
    //image.convertTo(image, CV_32F, 100.0 / 255.0); [0.255 -> 0.100]

    // Конвертация в градации серого и обратно в RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);

    int h = image.rows;
    int w = image.cols;

    // Изменение размера изображения
    int r_factor = (type == 4) ? 256 : 512;
    cv::resize(image, image, cv::Size(r_factor, r_factor));

    cv::Mat nchw_data = cv::dnn::blobFromImage(image, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);

    // Подготовка входных данных
    Ort::AllocatorWithDefaultOptions ortAllocator;
    auto inputName = ortSession->GetInputNameAllocated(0, ortAllocator);
    auto inputTypeInfo = ortSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto inputShape = inputTensorInfo.GetShape();

    auto outputName = ortSession->GetOutputNameAllocated(0, ortAllocator);
    auto outputTypeInfo = ortSession->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    auto outputShape = outputTensorInfo.GetShape();

    float inputTensorSize = 1;
    for (auto dim: inputShape) {
        inputTensorSize *= static_cast<float>(dim);
    }

    // Создание входного тензора
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                              (float *) nchw_data.data,
                                                              inputTensorSize,
                                                              inputShape.data(),
                                                              inputShape.size());

    auto bindings = Ort::IoBinding::IoBinding(*ortSession);

    bindings.BindInput(inputName.get(), input_tensor);
    bindings.BindOutput(outputName.get(), memoryInfo);

    Ort::RunOptions runOpts;
    ortSession->Run(runOpts, bindings);
    bindings.SynchronizeOutputs();

    // Получение результатов
    float *output_data = (float *) bindings.GetOutputValues()[0].GetTensorRawData();

    // Размер каждого канала
    size_t channelSize = static_cast<size_t>(r_factor * r_factor);
    std::vector<cv::Mat> channels;
    for (int i = 0; i < outputShape[1]; i++) {
        cv::Mat channel(r_factor, r_factor, CV_32F, const_cast<float *>(output_data + i * channelSize));
        channels.push_back(channel.clone());// clone гарантирует непрерывность данных
    }

    cv::UMat a(w, h, CV_32F);
    cv::UMat b(w, h, CV_32F);

    cv::resize(channels[0], a, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    cv::resize(channels[1], b, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

    // Объединение каналов L из исходного изображения и A, B из colorized
    cv::UMat result;
    std::vector<cv::UMat> merged_channels = {targetL, a, b};
    merge(merged_channels, result);

    cvtColor(result, result, cv::COLOR_Lab2BGR);
    result.convertTo(result, CV_8UC3, 255.0);
    result.copyTo(outimage);
}

int ColorSiggraph::process(const cv::Mat &inimage, cv::Mat &outimage) const {

    switch (type) {
        case 0: {
            process_Siggraph17(inimage, outimage);
        } break;
        case 1:
        case 2: {
            process_deoldify(inimage, outimage);
        } break;
        case 3:
        case 4: {
            process_DDColor(inimage, outimage);
        } break;
    }

    return 0;
}
