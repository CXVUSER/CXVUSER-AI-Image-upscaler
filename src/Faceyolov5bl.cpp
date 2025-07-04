#include "Faceyolov5bl.h"

static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
};

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].score;

    while (i <= j) {
        while (faceobjects[i].score > p)
            i++;

        while (faceobjects[j].score < p)
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

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
};

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
};

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
};

static void
generate_proposals(const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad, const ncnn::Mat &feat_blob,
                   float prob_threshold, std::vector<Object> &objects) {
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 15;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                const float *featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++) {
                    float score = featptr[15 + k];
                    if (score > class_score) {
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score);

                if (confidence >= prob_threshold) {
                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.score = confidence;
                    for (int l = 0; l < 3; l++) {
                        float x = featptr[2 * l + 5] * anchor_w + j * stride;
                        float y = featptr[2 * l + 1 + 5] * anchor_h + i * stride;
                        obj.pts.emplace_back(x, y);
                    }
                    objects.push_back(obj);
                }
            }
        }
    }
};

void Face_yolov5_bl::setThreshold(float prob_threshold_, float nms_threshold_) {
    this->prob_threshold = prob_threshold_;
    this->nms_threshold = nms_threshold_;
    return;
};

Face_yolov5_bl::Face_yolov5_bl() {
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 4;
};

Face_yolov5_bl::~Face_yolov5_bl() {
    net.clear();
};

int Face_yolov5_bl::Load(const std::wstring &model_path) {

    std::wstring model_p = model_path + L"/face_det/yolov5-blazeface.bin";
    std::wstring param_p = model_path + L"/face_det/yolov5-blazeface.param";

    {
        FILE *f = _wfopen(param_p.c_str(), L"rb");
        if (!f) {
            fwprintf(stderr, L"open param file %s failed\n", param_p.c_str());
            return -1;
        }

        int status = net.load_param(f);
        fclose(f);
        if (status != 0) {
            fwprintf(stderr, L"open param file %s failed\n", param_p.c_str());
            fclose(f);
            return -1;
        }
    }

    {
        FILE *f = _wfopen(model_p.c_str(), L"rb");
        if (!f) {
            fwprintf(stderr, L"open bin file %s failed\n", model_p.c_str());
            return -1;
        }

        int status = net.load_model(f);
        fclose(f);
        if (status != 0) {
            fwprintf(stderr, L"open bin file %s failed\n", model_p.c_str());
            fclose(f);
            return -1;
        }
    }

    return 0;
};

int Face_yolov5_bl::align_warp_face(const cv::Mat &img, Object_t &objects) {
    std::vector<cv::Point2f> face_template;
    face_template.push_back(cv::Point2f(192, 240));
    face_template.push_back(cv::Point2f(319, 240));
    face_template.push_back(cv::Point2f(257, 371));
    cv::Mat affine_matrix = cv::estimateAffinePartial2D(objects.pts, face_template);

    cv::Mat cropped_face;
    cv::warpAffine(img, cropped_face, affine_matrix, cv::Size(512, 512), 1, cv::BORDER_CONSTANT,
                   cv::Scalar(135, 133, 132));

    cv::Mat affine_matrix_inv;
    cv::invertAffineTransform(affine_matrix, affine_matrix_inv);

    affine_matrix_inv.copyTo(objects.trans_inv);
    cropped_face.copyTo(objects.trans_img);

    return 0;
};

int Face_yolov5_bl::Process(const cv::Mat &input_img, void *result) {

    const int target_size = 640;

    int img_w = input_img.cols;
    int img_h = input_img.rows;

    // letterbox pad to multiple of 32
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

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(input_img.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT,
                           0.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("data", in_pad);

    std::vector<Object> proposals;
    // stride 8
    {
        ncnn::Mat out;
        ex.extract("stride_8", out);

        ncnn::Mat anchors(6);
        anchors[0] = 5.f;
        anchors[1] = 6.f;
        anchors[2] = 10.f;
        anchors[3] = 13.f;
        anchors[4] = 21.f;
        anchors[5] = 26.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("stride_16", out);

        ncnn::Mat anchors(6);
        anchors[0] = 55.f;
        anchors[1] = 72.f;
        anchors[2] = 225.f;
        anchors[3] = 304.f;
        anchors[4] = 438.f;
        anchors[5] = 553.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }


    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    PipeResult_t *res = ((PipeResult_t *) result);
    res->face_count = count;
    for (int i = 0; i < count; i++) {
        Object fc = proposals[picked[i]];
        Object_t o;
        res->object.push_back(o);

        // adjust offset to original unpadded
        float x0 = (fc.rect.x - (wpad / 2)) / scale;
        float y0 = (fc.rect.y - (hpad / 2)) / scale;
        float x1 = (fc.rect.x + fc.rect.width - (wpad / 2)) / scale;
        float y1 = (fc.rect.y + fc.rect.height - (hpad / 2)) / scale;
        for (int j = 0; j < 3; j++) {
            float ptx = (fc.pts[j].x - (wpad / 2)) / scale;
            float pty = (fc.pts[j].y - (hpad / 2)) / scale;
            fc.pts[j] = cv::Point2f(ptx, pty);
        }
        for (size_t j = 0; j < 3; j++)
            res->object[i].pts.push_back(fc.pts[j]);

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        res->object[i].rect.x = x0;
        res->object[i].rect.y = y0;
        res->object[i].rect.width = x1 - x0;
        res->object[i].rect.height = y1 - y0;
    }

    for (int i = 0; i != res->face_count; ++i)
        align_warp_face(input_img, res->object[i]);

    return 0;
};

void Face_yolov5_bl::draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects) {

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];


        cv::circle(image, obj.pts[0], 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(image, obj.pts[1], 2, cv::Scalar(0, 255, 0), -1);
        cv::circle(image, obj.pts[2], 2, cv::Scalar(255, 0, 0), -1);
        cv::circle(image, obj.pts[3], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, obj.pts[4], 2, cv::Scalar(255, 255, 0), -1);
        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", "face", obj.score * 100);

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
    cv::waitKey();
};

void Face_yolov5_bl::Run(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor){};
void Face_yolov5_bl::PreProcess(const void *input_data, std::vector<Tensor_t> &input_tensor){};
void Face_yolov5_bl::PostProcess(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor, void *result){};