// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "include\retinaface.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <vector>

static inline float intersection_area(const FaceObject &a, const FaceObject &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
};

static void qsort_descent_inplace(std::vector<FaceObject> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
};

static void qsort_descent_inplace(std::vector<FaceObject> &faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
};

static void nms_sorted_bboxes(const std::vector<FaceObject> &faceobjects, std::vector<int> &picked, float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const FaceObject &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const FaceObject &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
};

// copy from src/layer/proposal.cpp
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales) {
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++) {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar);//round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++) {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float *anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
};

static void generate_proposals(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob, const ncnn::Mat &bbox_blob, const ncnn::Mat &landmark_blob, float prob_threshold, std::vector<FaceObject> &faceobjects) {
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q = 0; q < num_anchors; q++) {
        const float *anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q + num_anchors);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++) {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++) {
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold) {
                    // apply center size
                    float dx = bbox.channel(0)[index];
                    float dy = bbox.channel(1)[index];
                    float dw = bbox.channel(2)[index];
                    float dh = bbox.channel(3)[index];

                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float pb_cx = cx + anchor_w * dx;
                    float pb_cy = cy + anchor_h * dy;

                    float pb_w = anchor_w * exp(dw);
                    float pb_h = anchor_h * exp(dh);

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.landmark[0].x = cx + (anchor_w + 1) * landmark.channel(0)[index];
                    obj.landmark[0].y = cy + (anchor_h + 1) * landmark.channel(1)[index];
                    obj.landmark[1].x = cx + (anchor_w + 1) * landmark.channel(2)[index];
                    obj.landmark[1].y = cy + (anchor_h + 1) * landmark.channel(3)[index];
                    obj.landmark[2].x = cx + (anchor_w + 1) * landmark.channel(4)[index];
                    obj.landmark[2].y = cy + (anchor_h + 1) * landmark.channel(5)[index];
                    obj.landmark[3].x = cx + (anchor_w + 1) * landmark.channel(6)[index];
                    obj.landmark[3].y = cy + (anchor_h + 1) * landmark.channel(7)[index];
                    obj.landmark[4].x = cx + (anchor_w + 1) * landmark.channel(8)[index];
                    obj.landmark[4].y = cy + (anchor_h + 1) * landmark.channel(9)[index];
                    obj.prob = prob;

                    faceobjects.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
};

FaceR::FaceR(bool gpu, bool mnet) {
    face_template.push_back(cv::Point2f(192.98138, 239.94708));
    face_template.push_back(cv::Point2f(318.90277, 240.1936));
    face_template.push_back(cv::Point2f(256.63416, 314.01935));
    face_template.push_back(cv::Point2f(201.26117, 371.41043));
    face_template.push_back(cv::Point2f(313.08905, 371.15118));
    if (gpu)
        net_.opt.use_vulkan_compute = true;
    else
        net_.opt.use_vulkan_compute = false;

    this->gpu = gpu;
    this->mnet = mnet;
};

FaceR::~FaceR() {
    net_.clear();
};

void FaceR::setThreshold(float prob_threshold_, float nms_threshold_) {
    this->prob_threshold = prob_threshold_;
    this->nms_threshold = nms_threshold_;
    return;
};

int FaceR::Load(const std::wstring &model_path) {
    std::wstring model_param;
    std::wstring model_bin;

    if (true == mnet) {
        model_param = model_path + L"/face_det/mnet.25-opt.param";
        model_bin = model_path + L"/face_det/mnet.25-opt.bin";
    } else {
        model_param = model_path + L"/face_det/retinaface-R50.param";
        model_bin = model_path + L"/face_det/retinaface-R50.bin";
    }

    {
        FILE *f = _wfopen(model_param.c_str(), L"rb");
        if (!f) {
            fwprintf(stderr, L"open param file %s failed\n", model_param.c_str());
            return -1;
        }

        int status = net_.load_param(f);
        fclose(f);
        if (status != 0) {
            fwprintf(stderr, L"open param file %s failed\n", model_param.c_str());
            return -1;
        }
    }

    {
        FILE *f = _wfopen(model_bin.c_str(), L"rb");
        if (!f) {
            fwprintf(stderr, L"open bin file %s failed\n", model_bin.c_str());
            return -1;
        }

        int status = net_.load_model(f);
        fclose(f);
        if (status != 0) {
            fwprintf(stderr, L"open bin file %s failed\n", model_bin.c_str());
            return -1;
        }
    }
    return 0;
};

int FaceR::Process(const cv::Mat &bgr, void *result) {
    const int target_size = 640;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }
    //scale = float(target_size) / std::min(h, w);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    //int wpad = target_size - w;//(w + 31) / 32 * 32 - w;
    //int hpad = target_size - h;//(h + 31) / 32 * 32 - h;
    //ncnn::Mat in_pad;
    //ncnn::copy_make_border(in, in_pad, hpad, hpad - hpad, wpad, wpad - wpad, ncnn::BORDER_CONSTANT, 0.f);

    ncnn::Extractor ex = net_.create_extractor();

    ex.input("data", in);

    std::vector<FaceObject> faceproposals;

    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
        ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride32", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 32.f;
        scales[1] = 16.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, this->prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
        ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride16", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
        ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride8", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 2.f;
        scales[1] = 1.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();

    PipeResult_t *res = ((PipeResult_t *) result);
    res->face_count = face_count;
    for (int i = 0; i < face_count; i++) {
        FaceObject fc = faceproposals[picked[i]];
        Object_t o;
        res->object.push_back(o);

        // clip to image size
        float x0 = fc.rect.x / scale;
        float y0 = fc.rect.y / scale;
        float x1 = x0 + fc.rect.width / scale;
        float y1 = y0 + fc.rect.height / scale;

        x0 = std::max(std::min(x0, (float) img_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float) img_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float) img_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float) img_h - 1), 0.f);
        for (size_t j = 0; j < 5; j++) {
            //    -(wpad / 2) / scale
            float ptx = fc.landmark[j].x / scale;
            float pty = fc.landmark[j].y / scale;
            res->object[i].pts.push_back(cv::Point2f(ptx, pty));
        }

        res->object[i].rect.x = x0;
        res->object[i].rect.y = y0;
        res->object[i].rect.width = x1 - x0;
        res->object[i].rect.height = y1 - y0;
    }

    for (int i = 0; i != res->face_count; ++i)
        AlignFace(bgr, res->object[i]);

    return 0;
};

void FaceR::AlignFace(const cv::Mat &img, Object_t &objects) {

    cv::Mat affine_matrix = cv::estimateAffinePartial2D(objects.pts, face_template, cv::noArray(), cv::LMEDS);

    cv::Mat cropped_face;
    cv::warpAffine(img, cropped_face, affine_matrix, cv::Size(512, 512),
                   cv::InterpolationFlags::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(135, 133, 132));

    cv::Mat affine_matrix_inv;
    cv::invertAffineTransform(affine_matrix, affine_matrix_inv);

    affine_matrix_inv.copyTo(objects.trans_inv);
    cropped_face.copyTo(objects.trans_img);
};

void FaceR::draw_faceobjects(const cv::Mat &bgr, const std::vector<FaceObject> &faceobjects) {
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < faceobjects.size(); i++) {
        const FaceObject &obj = faceobjects[i];

        fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0));

        cv::circle(image, obj.landmark[0], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, obj.landmark[1], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, obj.landmark[2], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, obj.landmark[3], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, obj.landmark[4], 2, cv::Scalar(0, 255, 255), -1);

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
};

void FaceR::Run(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor){};
void FaceR::PreProcess(const void *input_data, std::vector<Tensor_t> &input_tensor){};
void FaceR::PostProcess(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor, void *result){};