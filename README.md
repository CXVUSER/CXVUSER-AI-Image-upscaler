# Upscayl-gfpgan-codeformer-realesr-ncnn-directml-vulkan 🚀

Ncnn with Vulkan implementation of **GFPGAN aims at developing Practical Algorithms for Real-world Face Restoration**

This repository contains the code and pre-trained models for a real-world face restoration algorithm based on the [GFPGAN](https://github.com/TencentARC/GFPGAN) method and optimized for mobile devices using the [NCNN](https://github.com/Tencent/ncnn) framework with a Vulkan backend.

The goal of this project is to develop practical algorithms that can restore the appearance of damaged or low-quality face images, such as those obtained from security cameras, old photographs, or social media profiles. The proposed approach combines the power of deep learning with the speed and efficiency of hardware acceleration, making it suitable for real-time applications on smartphones, drones, or robots.

## :construction: Model support :construction:

1. GFPGANCleanv1-NoCE-C2(NCNN)
2. GFPGAN 1.2,1.3,1.4(ONNX)
3. ncnn ESRGAN models
4. CodeFormer 0.1.0
5. RestoreFormer, RestoreFormer-plus-plus
6. GPEN

### Usage tips:
```
Usage: this_binary [options]...

 -i <img> path to image
 -s <digit> model scale factor (default=autodetect)
 -j <digit> custom output scale factor
 -t <digit> tile size (default=auto)
 -f restore faces (default=GFPGAN ncnn)
 -m <string> esrgan model name (default=./models/x4nomos8ksc)
 -g <string> gfpgan(or same as gfp) model path (default=./models/gfpgan_1.4.onnx)
 -x <digit> face detection threshold (default=0,5) (0,3..0,7 recommended)
 -c use CodeFormer face restore model
 -d swith face restore infer to onnx
 -w <digit> CodeFormer Fidelity (Only onnx) (default=0,7)
 -u Face Upsample (after face restore)
 -z <string> FaceUpsample model (ESRGAN)
 -p Use face parsing for accurate face masking (default=false)
 -o <string> override image output path
 -l <string> Face detector model (default=y7)
 -n no upsample
 -a wait
 -v verbose
```

### Sample:
```Console
background upsample by (4xNomos8kSC) and interpolate to 8x and face restored by codeformer(onnx) with fidelity 0,5 and upsample by high-fidelity-4x model
gfpgan-ncnn-vulkan.exe -i .\avatar6827912_4.jpeg -v -m ./models/4xNomos8kSC -x 0,3 -c -d -w 0,5 -u -z ./models/high-fidelity-4x -j 8
```

### Compile:
## Clone Project and Get Submodules

Make sure submodules are initialized and updated

```console
git clone https://github.com/CXVUSER/Upscayl-gpfgan-realesr-ncnn-directml.git
git submodule update --init --recursive
```

## Clone project with Submodules

```sh
git clone --recursive https://github.com/CXVUSER/Upscayl-gpfgan-realesr-ncnn-directml.git
```

## Project Prerequisites ⚙️

- CMake version 3.20 or later
- C++17 or above with filesystem support
- Clang-Tidy for code analysis (optional)
- Threads library
- Vulkan SDK
- glslangValidator executable
- OpenCV library
- OpenMP library
- ncnn library
- libwebp library

## Building 🛠️

Configure and build

```sh
mkdir build
cd build
cmake build ..
```
   
### References

1. <https://github.com/xinntao/Real-ESRGAN>
2. <https://github.com/TencentARC/GFPGAN>
3. <https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan>
4. <https://github.com/Tencent/ncnn>
5. <https://github.com/Tencent/ncnn/tree/master/tools/pnnx>
6. <https://github.com/pnnx/pnnx>
7. <https://github.com/derronqi/yolov8-face>
8. <https://github.com/FeiGeChuanShu/GFPGAN-ncnn>
9. <https://github.com/ultralytics/ultralytics>
10. <https://github.com/microsoft/DirectML>
11. <https://github.com/upscayl/upscayl-ncnn>
12. <https://github.com/sczhou/CodeFormer>
13. <https://github.com/microsoft/onnxruntime>

## Download Model files (CODEFORMER-GFPGAN-ESRGAN-ncnn-onnx model files)
https://github.com/CXVUSER/Upscayl-gpfgan-realesr-ncnn-directml/releases/download/1.0.2/models_1_0_2.7z
