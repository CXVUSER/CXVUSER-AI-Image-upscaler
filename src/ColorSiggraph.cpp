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
}

int ColorSiggraph::load(const wchar_t *modelpath) {
    if (gpu)
        net.opt.use_vulkan_compute = true;
    else
        net.opt.use_vulkan_compute = false;

    net.register_custom_layer("Sig17Slice", Sig17Slice_layer_creator);

    std::wstring model_param = modelpath;
    model_param += L"/COLOR/siggraph17_color_sim.param";
    std::wstring model_bin = modelpath;
    model_bin += L"/COLOR/siggraph17_color_sim.bin";

    FILE *f = _wfopen(model_param.c_str(), L"rb");
    if (f) {
        net.load_param(f);
        fclose(f);
    } else
        return 0;

    f = _wfopen(model_bin.c_str(), L"rb");
    if (f) {
        net.load_model(f);
        fclose(f);
    } else
        return 0;

    return 0;
}

int ColorSiggraph::process(const cv::Mat &inimage, cv::Mat &outimage) const {

    //fixed input size for the pretrained network
    const int W_in = 256;
    const int H_in = 256;
    cv::UMat lab, Base_img, L_u;
    cv::Mat input_img;
    Base_img = inimage.clone().getUMat(cv::ACCESS_RW);

    //normalize levels
    Base_img.convertTo(Base_img, CV_32F, 1.0 / 255);

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
    return 0;
}
