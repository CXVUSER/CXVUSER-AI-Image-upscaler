// codeformer implemented with ncnn library

#ifndef CODEFORMER_H
#define CODEFORMER_H
#include "include/model.h"
#include <memory>

class Encoder;
class Generator;
class CodeFormer {
public:
    CodeFormer(bool gpu = true);
    ~CodeFormer();

    int Load(const std::wstring &model_path);
    int Process(const cv::Mat &input_img, CodeFormerResult_t &model_result);

private:
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Generator> generator_;
    bool gpu;
};
#endif// CODEFORMER_H