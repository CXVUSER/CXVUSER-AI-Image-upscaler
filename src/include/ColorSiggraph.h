#ifndef COLORNET_H_INCLUDED
#define COLORNET_H_INCLUDED
#include "include/model.h"

class ColorSiggraph {
public:
    ColorSiggraph(bool gpu = true);
    ~ColorSiggraph();

    int load(const wchar_t *modelpath);
    int process(const cv::Mat &inimage, cv::Mat &outimage) const;

public:

private:
    ncnn::Net net;
    bool gpu;
};


#endif// COLORNET_H_INCLUDED