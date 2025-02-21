// realesrgan implemented with ncnn library

#ifndef REALESRGAN_H
#define REALESRGAN_H

#include <string>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class RealESRGAN
{
public:
    RealESRGAN(bool gpu = true,bool tta_mode = false);
    ~RealESRGAN();

    int load(const wchar_t *parampath, const wchar_t *modelpath);
    int process(const ncnn::Mat &inimage, ncnn::Mat &outimage) const;
    int process_gpu(const ncnn::Mat &inimage, ncnn::Mat &outimage) const;
    int process_cpu(const ncnn::Mat &inimage, ncnn::Mat &outimage) const;

public:
    // realesrgan parameters
    int scale;
    int tilesize;
    int prepadding;

private:
    ncnn::Net net;
    ncnn::Pipeline *realesrgan_preproc = nullptr;
    ncnn::Pipeline *realesrgan_postproc = nullptr;
    ncnn::Layer *bicubic_2x = nullptr;
    ncnn::Layer *bicubic_3x = nullptr;
    ncnn::Layer *bicubic_4x = nullptr;
    bool tta_mode;
    bool gpu;
};

#endif // REALESRGAN_H
