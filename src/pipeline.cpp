// codeformer implemented with ncnn library

#include "include/pipeline.h"
namespace wsdsb {

    PipeLine::PipeLine(int scale) : codeformer_(new CodeFormer), face_detector_(new FaceG(scale)) {
    }
    PipeLine::~PipeLine() {
        codeformer_.reset();
        face_detector_.reset();
    }

    static void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample) {
        trans_matrix_inv.at<float>(0, 2) += 1.0;
        trans_matrix_inv.at<float>(1, 2) += 1.0;

        cv::Mat inv_restored;
        cv::warpAffine(restored_face, inv_restored, trans_matrix_inv, bg_upsample.size(), 1, 0);

        cv::Mat mask = cv::Mat::ones(cv::Size(512, 512), CV_8UC1) * 255;
        cv::Mat inv_mask;
        cv::warpAffine(mask, inv_mask, trans_matrix_inv, bg_upsample.size(), 1, 0);

        cv::Mat inv_mask_erosion;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4));
        cv::erode(inv_mask, inv_mask_erosion, kernel);
        cv::Mat pasted_face;
        cv::bitwise_and(inv_restored, inv_restored, pasted_face, inv_mask_erosion);

        int total_face_area = cv::countNonZero(inv_mask_erosion);
        int w_edge = int(std::sqrt(total_face_area) / 20);
        int erosion_radius = w_edge * 2;
        cv::Mat inv_mask_center;
        kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erosion_radius, erosion_radius));
        cv::erode(inv_mask_erosion, inv_mask_center, kernel);

        int blur_size = w_edge * 2;
        cv::Mat inv_soft_mask;
        cv::GaussianBlur(inv_mask_center, inv_soft_mask, cv::Size(blur_size + 1, blur_size + 1), 0, 0, 4);

        cv::Mat inv_soft_mask_f;
        inv_soft_mask.convertTo(inv_soft_mask_f, CV_32F, 1 / 255.f, 0.f);

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
        pipeline_config_ = pipeline_config;
        if (false == pipeline_config.onnx) {
            int ret = codeformer_->Load(pipeline_config_.model_path);
            if (ret < 0) {
                return -1;
            }
        }
        int ret = face_detector_->Load(pipeline_config.model_path);
        if (ret < 0) {
            return -1;
        }

        return 0;
    }
    int PipeLine::Apply(const cv::Mat &input_img, cv::Mat &output_img) {
        PipeResult_t pipe_result;
        fprintf(stderr, "Detecting faces...\n");
        face_detector_->Process(input_img, (void *) &pipe_result);
        fprintf(stderr, "Detected %d faces\n", pipe_result.face_count);

        for (int i = 0; i != pipe_result.face_count; ++i) {
            fprintf(stderr, "Codeformer process...\n");
            //codeformer_->Process(pipe_result.object[i].trans_img, pipe_result.codeformer_result[i]);
            if (pipeline_config_.onnx) {
                char d[_MAX_PATH];
                sprintf(d, "%.1f", pipeline_config_.w);
                *strrchr(d, ',') = '.';

                std::stringstream str;
                str << pipeline_config_.name << "_" << i + 1 << "_crop.png" << std::ends;
                cv::imwrite(str.view().data(), pipe_result.object[i].trans_img);
                std::stringstream str3;
                str3 << pipeline_config_.name << "_" << i + 1 << "_" << pipeline_config_.w << "_codeformer_crop.png" << std::ends;

                std::stringstream str2;
                str2 << "python codeformer_onnx.py --model_path "
                     << "./codeformer.onnx"
                     << " --image_path " << str.view().data() << " --w " << d << std::ends;
                system(str2.view().data());

                cv::Mat restored_face = cv::imread("output.jpg", 1);
                cv::imwrite(str3.view().data(), restored_face);
                paste_faces_to_input_image(restored_face, pipe_result.object[i].trans_inv, output_img);
            }
            if (pipeline_config_.ncnn) {
                codeformer_->Process(pipe_result.object[i].trans_img, pipe_result.codeformer_result[i]);
                paste_faces_to_input_image(pipe_result.codeformer_result[i].restored_face, pipe_result.object[i].trans_inv, output_img);
            }
        }

        return 0;
    }
}// namespace wsdsb