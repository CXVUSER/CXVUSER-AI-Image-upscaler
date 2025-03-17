#ifndef COLORNET_H_INCLUDED
#define COLORNET_H_INCLUDED
#include "include/helpers.h"
#include "include/model.h"

class ColorSiggraph {
public:
    ColorSiggraph(bool gpu = true);
    ~ColorSiggraph();

    int load(Ort::SessionOptions &sessOpt, std::wstring &model);
    int process(const cv::Mat &inimage, cv::Mat &outimage) const;
    void process_Siggraph17(const cv::Mat &inimage, cv::Mat &outimage) const;
    void process_deoldify(const cv::Mat &inimage, cv::Mat &outimage) const;
    void process_DDColor(const cv::Mat &inimage, cv::Mat &outimage) const;


public:
private:
    ncnn::Net net;
    bool gpu;
    int type;
    Ort::Env *env = nullptr;
    Ort::Session *ortSession = nullptr;
};
#endif// COLORNET_H_INCLUDED