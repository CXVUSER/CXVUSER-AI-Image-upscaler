// codeformer implemented with ncnn library

#include "include/pipeline.h"
#include "realesrgan.h"
//#include "wic_image.h"
extern unsigned char *wic_decode_image(const wchar_t *filepath, int *w, int *h, int *c);
extern int wic_encode_image(const wchar_t *filepath, int w, int h, int c, void *bgrdata);

namespace wsdsb {

    PipeLine::PipeLine() {
    }
    PipeLine::~PipeLine() {
        if (!pipeline_config_.onnx)
            delete codeformer_;
        delete face_detector_;
    }

    static void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample, PipelineConfig_t &pipe) {
        trans_matrix_inv.at<double>(0, 2) += (double) pipe.model_scale;
        trans_matrix_inv.at<double>(1, 2) += (double) pipe.model_scale;
        cv::Mat restored_face_up;
        double ups_f = 0.0;
        if (pipe.face_upsample && !pipe.up_model.empty()) {
            ups_f = 4.0;
            fprintf(stderr, "Upsample face start...\n");
            RealESRGAN real_esrgan;
            real_esrgan.scale = 4;
            real_esrgan.prepadding = 10;
            real_esrgan.tilesize = 200;

            std::wstringstream str_param;
            str_param << pipe.up_model << ".param" << std::ends;
            std::wstringstream str_bin;
            str_bin << pipe.up_model << ".bin" << std::ends;

            real_esrgan.load(str_param.view().data(), str_bin.view().data());

            int w{}, h{}, c{};
#if _WIN32
            void *pixeldata = wic_decode_image(L"output.jpg", &w, &h, &c);
#else
#endif
            ncnn::Mat bg_presample(w, h, (void *) pixeldata, (size_t) c, c);
            ncnn::Mat bg_upsamplencnn(w * 4, h * 4, (size_t) c, c);
            std::wstringstream str_param1;
            str_param1 << "out.png" << std::ends;
            real_esrgan.process(bg_presample, bg_upsamplencnn);
#if _WIN32
            wic_encode_image(str_param1.view().data(), w * 4, h * 4, 3, bg_upsamplencnn.data);
#else
#endif

            restored_face_up = cv::imread("out.png", 1);

            trans_matrix_inv /= ups_f;
            trans_matrix_inv.at<double>(0, 2) *= ups_f;
            trans_matrix_inv.at<double>(1, 2) *= ups_f;

            fprintf(stderr, "Upsample face finish...\n");
        }

        cv::Mat inv_restored;
        if (pipe.face_upsample)
            cv::warpAffine(restored_face_up, inv_restored, trans_matrix_inv, bg_upsample.size(), 1, 0);
        else
            cv::warpAffine(restored_face, inv_restored, trans_matrix_inv, bg_upsample.size(), 1, 0);

        cv::Size ms_size;

        if (pipe.face_upsample)
            ms_size = cv::Size(512 * ups_f, 512 * ups_f);
        else
            ms_size = cv::Size(512, 512);

        cv::Mat mask = cv::Mat::ones(ms_size, CV_8UC1) * 255;
        cv::Mat inv_mask;
        cv::warpAffine(mask, inv_mask, trans_matrix_inv, bg_upsample.size(), 1, 0);


        cv::Size krn_size;
        if (pipe.face_upsample)
            krn_size = cv::Size(4 * ups_f, 4 * ups_f);
        else
            krn_size = cv::Size(4, 4);

        cv::Mat inv_mask_erosion;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, krn_size);
        cv::erode(inv_mask, inv_mask_erosion, kernel);
        cv::Mat pasted_face;
        cv::bitwise_and(inv_restored, inv_restored, pasted_face, inv_mask_erosion);

        int total_face_area = cv::countNonZero(inv_mask_erosion);
        int w_edge = int(std::sqrt(total_face_area) / 20);
        int erosion_radius = w_edge * 2;
        cv::Mat inv_mask_center;
        kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erosion_radius, erosion_radius));
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
            codeformer_ = new CodeFormer();
            int ret = codeformer_->Load(pipeline_config_.model_path);
            if (ret < 0) {
                return -1;
            }
        }

        if (pipeline_config.custom_scale)
            face_detector_ = new FaceG(pipeline_config.custom_scale, pipeline_config.prob_thr, pipeline_config.nms_thr);
        else
            face_detector_ = new FaceG(pipeline_config.model_scale, pipeline_config.prob_thr, pipeline_config.nms_thr);

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

        char d[_MAX_PATH];
        sprintf(d, "%.1f", pipeline_config_.w);
        *strrchr(d, ',') = '.';

        for (int i = 0; i != pipe_result.face_count; ++i) {
            fprintf(stderr, "Codeformer process %d face...\n", i + 1);

            std::stringstream str;
            str << pipeline_config_.name << "_" << i + 1 << "_crop.png" << std::ends;
            cv::imwrite(str.view().data(), pipe_result.object[i].trans_img);
            std::stringstream str3;
            str3 << pipeline_config_.name << "_" << i + 1 << "_" << pipeline_config_.w << "_codeformer_crop.png" << std::ends;
            //codeformer_->Process(pipe_result.object[i].trans_img, pipe_result.codeformer_result[i]);
            if (pipeline_config_.onnx) {

                std::stringstream str2;
                str2 << "python codeformer_onnx.py --model_path "
                     << pipeline_config_.model_path << "codeformer_0_1_0.onnx"
                     << " --image_path " << str.view().data() << " --w " << d << std::ends;
                system(str2.view().data());

                cv::Mat restored_face = cv::imread("output.jpg", 1);
                cv::imwrite(str3.view().data(), restored_face);
                fprintf(stderr, "Paste %d face in photo...\n", i + 1);
                paste_faces_to_input_image(restored_face, pipe_result.object[i].trans_inv, output_img,
                                           pipeline_config_);
            }
            if (pipeline_config_.ncnn) {
                codeformer_->Process(pipe_result.object[i].trans_img, pipe_result.codeformer_result[i]);
                cv::imwrite(str3.view().data(), pipe_result.codeformer_result[i].restored_face);
                fprintf(stderr, "Paste %d face in photo...\n", i + 1);
                paste_faces_to_input_image(pipe_result.codeformer_result[i].restored_face, pipe_result.object[i].trans_inv, output_img,
                                           pipeline_config_);
            }
        }

        return 0;
    }
}// namespace wsdsb