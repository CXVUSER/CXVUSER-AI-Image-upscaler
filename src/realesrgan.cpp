// realesrgan implemented with ncnn library

#include "realesrgan.h"

#include <algorithm>
#include <vector>

static const uint32_t realesrgan_preproc_spv_data[] = {
#include "realesrgan_preproc.spv.hex.h"
};
static const uint32_t realesrgan_preproc_fp16s_spv_data[] = {
#include "realesrgan_preproc_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_preproc_int8s_spv_data[] = {
#include "realesrgan_preproc_int8s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_spv_data[] = {
#include "realesrgan_postproc.spv.hex.h"
};
static const uint32_t realesrgan_postproc_fp16s_spv_data[] = {
#include "realesrgan_postproc_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_int8s_spv_data[] = {
#include "realesrgan_postproc_int8s.spv.hex.h"
};

static const uint32_t realesrgan_preproc_tta_spv_data[] = {
#include "realesrgan_preproc_tta.spv.hex.h"
};
static const uint32_t realesrgan_preproc_tta_fp16s_spv_data[] = {
#include "realesrgan_preproc_tta_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_preproc_tta_int8s_spv_data[] = {
#include "realesrgan_preproc_tta_int8s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_tta_spv_data[] = {
#include "realesrgan_postproc_tta.spv.hex.h"
};
static const uint32_t realesrgan_postproc_tta_fp16s_spv_data[] = {
#include "realesrgan_postproc_tta_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_tta_int8s_spv_data[] = {
#include "realesrgan_postproc_tta_int8s.spv.hex.h"
};

RealESRGAN::RealESRGAN(bool gpu, bool _tta_mode) {
    if (true == gpu) {
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_int8_storage = true;
        net.opt.use_int8_arithmetic = false;

        net.set_vulkan_device(0);
    } else {
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_int8_storage = true;
        net.set_vulkan_device(0);
    }

    realesrgan_preproc = 0;
    realesrgan_postproc = 0;
    bicubic_2x = 0;
    bicubic_3x = 0;
    bicubic_4x = 0;
    tta_mode = _tta_mode;
    this->gpu = gpu;
};

RealESRGAN::~RealESRGAN() {
    // cleanup preprocess and postprocess pipeline
    {
        if (realesrgan_preproc)
            delete realesrgan_preproc;

        if (realesrgan_postproc)
            delete realesrgan_postproc;
    }

    if (bicubic_2x) {
        bicubic_2x->destroy_pipeline(net.opt);
        delete bicubic_2x;
    }

    if (bicubic_3x) {
        bicubic_3x->destroy_pipeline(net.opt);
        delete bicubic_3x;
    }

    if (bicubic_4x) {
        bicubic_4x->destroy_pipeline(net.opt);
        delete bicubic_4x;
    }
};

int RealESRGAN::load(const wchar_t *parampath, const wchar_t *modelpath) {
    {
        FILE *fp = _wfopen(parampath, L"rb");
        if (!fp) {
            fwprintf(stderr, L"🚨 Error: Failed to open %ls\n", parampath);
            return -1;
        }

        int status = net.load_param(fp);
        fclose(fp);
        if (status != 0) {
            fwprintf(stderr, L"🚨 Error: Failed to open %ls\n", parampath);
            return -1;
        }
    }
    {
        FILE *fp = _wfopen(modelpath, L"rb");
        if (!fp) {
            fwprintf(stderr, L"🚨 Error: Failed to open %ls\n", modelpath);
            return -1;
        }

        int status = net.load_model(fp);
        fclose(fp);
        if (status != 0) {
            fwprintf(stderr, L"🚨 Error: Failed to open %ls\n", parampath);
            return -1;
        }
    }

    createSpv();

    // bicubic 2x/3x/4x for alpha channel
    {
        bicubic_2x = ncnn::create_layer("Interp");
        bicubic_2x->vkdev = net.vulkan_device();

        ncnn::ParamDict pd;
        pd.set(0, 3);// bicubic
        pd.set(1, 2.f);
        pd.set(2, 2.f);
        bicubic_2x->load_param(pd);

        bicubic_2x->create_pipeline(net.opt);
    }
    {
        bicubic_3x = ncnn::create_layer("Interp");
        bicubic_3x->vkdev = net.vulkan_device();

        ncnn::ParamDict pd;
        pd.set(0, 3);// bicubic
        pd.set(1, 3.f);
        pd.set(2, 3.f);
        bicubic_3x->load_param(pd);

        bicubic_3x->create_pipeline(net.opt);
    }
    {
        bicubic_4x = ncnn::create_layer("Interp");
        bicubic_4x->vkdev = net.vulkan_device();

        ncnn::ParamDict pd;
        pd.set(0, 3);// bicubic
        pd.set(1, 4.f);
        pd.set(2, 4.f);
        bicubic_4x->load_param(pd);

        bicubic_4x->create_pipeline(net.opt);
    }

    return 0;
};

int RealESRGAN::process(const ncnn::Mat &inimage, ncnn::Mat &outimage) const {
    if (gpu)
        return process_spv(inimage, outimage);
    else
        return process_no_spv(inimage, outimage);
};

int RealESRGAN::process_spv(const ncnn::Mat &inimage, ncnn::Mat &outimage) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    ncnn::VkAllocator *blob_vkallocator = net.vulkan_device()->acquire_blob_allocator();
    ncnn::VkAllocator *staging_vkallocator = net.vulkan_device()->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // each tile 100x100
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    // #pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, h);

        ncnn::Mat in;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char *) pixeldata + in_tile_y0 * w * channels, (size_t) channels, 1);
        } else {
            if (channels == 3) {
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));
            }
            if (channels == 4) {
                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGRA2RGBA, w, (in_tile_y1 - in_tile_y0));
            }
        }

        ncnn::VkCompute cmd(net.vulkan_device());

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage) {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t) channels, 1, blob_vkallocator);
        } else {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t) 4u, 1, blob_vkallocator);
        }

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            if (tta_mode) {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];
                    bindings[9] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realesrgan_preproc, bindings, constants, dispatcher);
                }

                // realesrgan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("data", in_tile_gpu[ti]);

                    ex.extract("output", out_tile_gpu[ti], cmd);

                    {
                        cmd.submit_and_wait();
                        cmd.reset();
                    }
                }

                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                {
                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = out_tile_gpu[0];
                    bindings[1] = out_tile_gpu[1];
                    bindings[2] = out_tile_gpu[2];
                    bindings[3] = out_tile_gpu[3];
                    bindings[4] = out_tile_gpu[4];
                    bindings[5] = out_tile_gpu[5];
                    bindings[6] = out_tile_gpu[6];
                    bindings[7] = out_tile_gpu[7];
                    bindings[8] = out_alpha_tile_gpu;
                    bindings[9] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = out_tile_gpu[0].w;
                    constants[1].i = out_tile_gpu[0].h;
                    constants[2].i = out_tile_gpu[0].cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = prepadding * scale;
                    constants[9].i = prepadding * scale;
                    constants[10].i = channels;
                    constants[11].i = out_alpha_tile_gpu.w;
                    constants[12].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realesrgan_postproc, bindings, constants, dispatcher);
                }
            } else {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                    if (channels == 4) {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realesrgan_preproc, bindings, constants, dispatcher);
                }

                // realesrgan
                ncnn::VkMat out_tile_gpu;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("data", in_tile_gpu);

                    ex.extract("output", out_tile_gpu, cmd);
                }

                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2) {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 3) {
                        bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                {
                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = out_tile_gpu;
                    bindings[1] = out_alpha_tile_gpu;
                    bindings[2] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = out_tile_gpu.w;
                    constants[1].i = out_tile_gpu.h;
                    constants[2].i = out_tile_gpu.cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = prepadding * scale;
                    constants[9].i = prepadding * scale;
                    constants[10].i = channels;
                    constants[11].i = out_alpha_tile_gpu.w;
                    constants[12].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realesrgan_postproc, bindings, constants, dispatcher);
                }
            }

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }

            fprintf(stderr, "%.2f%%\n", (float) (yi * xtiles + xi) / (ytiles * xtiles) * 100);
        }

        // download
        {
            ncnn::Mat out;

            if (opt.use_fp16_storage && opt.use_int8_storage) {
                out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char *) outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, (size_t) channels, 1);
            }

            cmd.record_clone(out_gpu, out, opt);

            cmd.submit_and_wait();

            if (!(opt.use_fp16_storage && opt.use_int8_storage)) {
                if (channels == 3) {
                    out.to_pixels((unsigned char *) outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGB2BGR);
                }
                if (channels == 4) {
                    out.to_pixels((unsigned char *) outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGBA2BGRA);
                }
            }
        }
    }

    net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
    net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);

    return 0;
};

int RealESRGAN::process_no_spv(const ncnn::Mat &inimage, ncnn::Mat &outimage) const {
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    ncnn::Option opt = net.opt;

    // each tile 100x100
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, h);

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int in_tile_x0 = std::max(xi * TILE_SIZE_X - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * TILE_SIZE_X + prepadding, w);

            // crop tile
            ncnn::Mat in;
            {
                if (channels == 3) {
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGR2RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
                if (channels == 4) {
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_BGRA2RGBA, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
            }

            ncnn::Mat out;

            if (tta_mode) {
                // split alpha and preproc
                ncnn::Mat in_tile[8];
                ncnn::Mat in_alpha_tile;
                {
                    in_tile[0].create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr0 = in_tile[0].channel(q);

                        for (int i = 0; i < in.h; i++) {
                            for (int j = 0; j < in.w; j++) {
                                *outptr0++ = *ptr++ * (1 / 255.f);
                            }
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * TILE_SIZE_Y + prepadding - h, prepadding), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding - w, prepadding), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile[0], in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile[0] = in_tile_padded;
                }

                // the other 7 directions
                {
                    in_tile[1].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[2].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[3].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[4].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[5].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[6].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[7].create(in_tile[0].h, in_tile[0].w, 3);

                    for (int q = 0; q < 3; q++) {
                        const ncnn::Mat in_tile_0 = in_tile[0].channel(q);
                        ncnn::Mat in_tile_1 = in_tile[1].channel(q);
                        ncnn::Mat in_tile_2 = in_tile[2].channel(q);
                        ncnn::Mat in_tile_3 = in_tile[3].channel(q);
                        ncnn::Mat in_tile_4 = in_tile[4].channel(q);
                        ncnn::Mat in_tile_5 = in_tile[5].channel(q);
                        ncnn::Mat in_tile_6 = in_tile[6].channel(q);
                        ncnn::Mat in_tile_7 = in_tile[7].channel(q);

                        for (int i = 0; i < in_tile[0].h; i++) {
                            const float *outptr0 = in_tile_0.row(i);
                            float *outptr1 = in_tile_1.row(in_tile[0].h - 1 - i);
                            float *outptr2 = in_tile_2.row(i) + in_tile[0].w - 1;
                            float *outptr3 = in_tile_3.row(in_tile[0].h - 1 - i) + in_tile[0].w - 1;

                            for (int j = 0; j < in_tile[0].w; j++) {
                                float *outptr4 = in_tile_4.row(j) + i;
                                float *outptr5 = in_tile_5.row(in_tile[0].w - 1 - j) + i;
                                float *outptr6 = in_tile_6.row(j) + in_tile[0].h - 1 - i;
                                float *outptr7 = in_tile_7.row(in_tile[0].w - 1 - j) + in_tile[0].h - 1 - i;

                                float v = *outptr0++;

                                *outptr1++ = v;
                                *outptr2-- = v;
                                *outptr3-- = v;
                                *outptr4 = v;
                                *outptr5 = v;
                                *outptr6 = v;
                                *outptr7 = v;
                            }
                        }
                    }
                }

                // realsr
                ncnn::Mat out_tile[8];
                for (int ti = 0; ti < 8; ti++) {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("data", in_tile[ti]);

                    ex.extract("output", out_tile[ti]);
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile = in_alpha_tile;
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    for (int q = 0; q < 3; q++) {
                        const ncnn::Mat out_tile_0 = out_tile[0].channel(q);
                        const ncnn::Mat out_tile_1 = out_tile[1].channel(q);
                        const ncnn::Mat out_tile_2 = out_tile[2].channel(q);
                        const ncnn::Mat out_tile_3 = out_tile[3].channel(q);
                        const ncnn::Mat out_tile_4 = out_tile[4].channel(q);
                        const ncnn::Mat out_tile_5 = out_tile[5].channel(q);
                        const ncnn::Mat out_tile_6 = out_tile[6].channel(q);
                        const ncnn::Mat out_tile_7 = out_tile[7].channel(q);
                        float *outptr = out.channel(q);

                        for (int i = 0; i < out.h; i++) {
                            const float *ptr0 = out_tile_0.row(i + prepadding * scale) + prepadding * scale;
                            const float *ptr1 = out_tile_1.row(out_tile[0].h - 1 - i - prepadding * scale) + prepadding * scale;
                            const float *ptr2 = out_tile_2.row(i + prepadding * scale) + out_tile[0].w - 1 - prepadding * scale;
                            const float *ptr3 = out_tile_3.row(out_tile[0].h - 1 - i - prepadding * scale) + out_tile[0].w - 1 - prepadding * scale;

                            for (int j = 0; j < out.w; j++) {
                                const float *ptr4 = out_tile_4.row(j + prepadding * scale) + i + prepadding * scale;
                                const float *ptr5 = out_tile_5.row(out_tile[0].w - 1 - j - prepadding * scale) + i + prepadding * scale;
                                const float *ptr6 = out_tile_6.row(j + prepadding * scale) + out_tile[0].h - 1 - i - prepadding * scale;
                                const float *ptr7 = out_tile_7.row(out_tile[0].w - 1 - j - prepadding * scale) + out_tile[0].h - 1 - i - prepadding * scale;

                                float v = (*ptr0++ + *ptr1++ + *ptr2-- + *ptr3-- + *ptr4 + *ptr5 + *ptr6 + *ptr7) / 8;

                                *outptr++ = v * 255.f + 0.5f;
                            }
                        }
                    }

                    if (channels == 4) {
                        memcpy(out.channel_range(3, 1), out_alpha_tile, out_alpha_tile.total() * sizeof(float));
                    }
                }
            } else {
                // split alpha and preproc
                ncnn::Mat in_tile;
                ncnn::Mat in_alpha_tile;
                {
                    in_tile.create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr = in_tile.channel(q);

                        for (int i = 0; i < in.w * in.h; i++) {
                            *outptr++ = *ptr++ * (1 / 255.f);
                        }
                    }

                    if (channels == 4) {
                        in_alpha_tile = in.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * TILE_SIZE_Y + prepadding - h, prepadding), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding - w, prepadding), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile = in_tile_padded;
                }

                // realsr
                ncnn::Mat out_tile;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("data", in_tile);

                    ex.extract("output", out_tile);
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4) {
                    if (scale == 1) {
                        out_alpha_tile = in_alpha_tile;
                    }
                    if (scale == 4) {
                        bicubic_4x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    for (int q = 0; q < 3; q++) {
                        float *outptr = out.channel(q);

                        for (int i = 0; i < out.h; i++) {
                            const float *ptr = out_tile.channel(q).row(i + prepadding * scale) + prepadding * scale;

                            for (int j = 0; j < out.w; j++) {
                                *outptr++ = *ptr++ * 255.f + 0.5f;
                            }
                        }
                    }

                    if (channels == 4) {
                        memcpy(out.channel_range(3, 1), out_alpha_tile, out_alpha_tile.total() * sizeof(float));
                    }
                }
            }

            {
                if (channels == 3) {
                    out.to_pixels((unsigned char *) outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels + xi * scale * TILE_SIZE_X * channels, ncnn::Mat::PIXEL_RGB2BGR, w * scale * channels);
                }
                if (channels == 4) {
                    out.to_pixels((unsigned char *) outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels + xi * scale * TILE_SIZE_X * channels, ncnn::Mat::PIXEL_RGBA2BGRA, w * scale * channels);
                }
            }
        }
    }

    return 0;
};

void RealESRGAN::enableTTA(bool enable) {
    this->tta_mode = enable;

    if (realesrgan_preproc)
        delete realesrgan_preproc;

    if (realesrgan_postproc)
        delete realesrgan_postproc;

    createSpv();
};

void RealESRGAN::createSpv() {
    if (true == gpu) {
        // initialize preprocess and postprocess pipeline
        {
            std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
            specializations[0].i = 1;
#else
            specializations[0].i = 0;
#endif

            realesrgan_preproc = new ncnn::Pipeline(net.vulkan_device());
            realesrgan_preproc->set_optimal_local_size_xyz(32, 32, 3);

            realesrgan_postproc = new ncnn::Pipeline(net.vulkan_device());
            realesrgan_postproc->set_optimal_local_size_xyz(32, 32, 3);

            if (tta_mode) {
                if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                    realesrgan_preproc->create(realesrgan_preproc_tta_int8s_spv_data, sizeof(realesrgan_preproc_tta_int8s_spv_data), specializations);
                else if (net.opt.use_fp16_storage)
                    realesrgan_preproc->create(realesrgan_preproc_tta_fp16s_spv_data, sizeof(realesrgan_preproc_tta_fp16s_spv_data), specializations);
                else
                    realesrgan_preproc->create(realesrgan_preproc_tta_spv_data, sizeof(realesrgan_preproc_tta_spv_data), specializations);

                if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                    realesrgan_postproc->create(realesrgan_postproc_tta_int8s_spv_data, sizeof(realesrgan_postproc_tta_int8s_spv_data), specializations);
                else if (net.opt.use_fp16_storage)
                    realesrgan_postproc->create(realesrgan_postproc_tta_fp16s_spv_data, sizeof(realesrgan_postproc_tta_fp16s_spv_data), specializations);
                else
                    realesrgan_postproc->create(realesrgan_postproc_tta_spv_data, sizeof(realesrgan_postproc_tta_spv_data), specializations);
            } else {
                if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                    realesrgan_preproc->create(realesrgan_preproc_int8s_spv_data, sizeof(realesrgan_preproc_int8s_spv_data), specializations);
                else if (net.opt.use_fp16_storage)
                    realesrgan_preproc->create(realesrgan_preproc_fp16s_spv_data, sizeof(realesrgan_preproc_fp16s_spv_data), specializations);
                else
                    realesrgan_preproc->create(realesrgan_preproc_spv_data, sizeof(realesrgan_preproc_spv_data), specializations);

                if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                    realesrgan_postproc->create(realesrgan_postproc_int8s_spv_data, sizeof(realesrgan_postproc_int8s_spv_data), specializations);
                else if (net.opt.use_fp16_storage)
                    realesrgan_postproc->create(realesrgan_postproc_fp16s_spv_data, sizeof(realesrgan_postproc_fp16s_spv_data), specializations);
                else
                    realesrgan_postproc->create(realesrgan_postproc_spv_data, sizeof(realesrgan_postproc_spv_data), specializations);
            }
        }
    }
};