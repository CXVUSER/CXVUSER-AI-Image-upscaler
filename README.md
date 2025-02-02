# Upscayl-gfpgan-realesr-ncnn-directml 🚀

Ncnn with Vulkan implementation of **GFPGAN aims at developing Practical Algorithms for Real-world Face Restoration**

This repository contains the code and pre-trained models for a real-world face restoration algorithm based on the [GFPGAN](https://github.com/TencentARC/GFPGAN) method and optimized for mobile devices using the [NCNN](https://github.com/Tencent/ncnn) framework with a Vulkan backend.

The goal of this project is to develop practical algorithms that can restore the appearance of damaged or low-quality face images, such as those obtained from security cameras, old photographs, or social media profiles. The proposed approach combines the power of deep learning with the speed and efficiency of hardware acceleration, making it suitable for real-time applications on smartphones, drones, or robots.

### Usage tips:
```
Usage: this_binary [options]...

 -i <img> path to image
 -s <digit> scale factor (default=4)
 -t <digit> tile size (default=auto)
 -f restore faces (GFPGAN 1.4) (default=0)
 -m <string> esrgan model name (default=./models/x4nomos8ksc)
 -g <string> gfpgan model path (default=./models/gfpgan_1.4)
 -x <digit> YOLOV5 face detection threshold (default=0,5) (0.3..0.7 recommended)
 -c use gfpgan-ncnn infer instead of onnx(DirectML or CUDA prefer) (only GFPGANCleanv1-NoCE-C2 model and CPU backend)
 -n no upsample
 -v verbose
```

### Sample:
```Console
gfpgan-ncnn-vulkan.exe -i .\avatar6827912_4.jpeg -v -f -x 0,3
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
mkdir -p build && cd build
cmake ..
cmake --build . --parallel $(($(nproc) - 1))
```

## :construction: Model support :construction:

1. GFPGANCleanv1-NoCE-C2
2. GFPGAN 1.2,1.3,1.4
3. ncnn esrgan models
   
### References

1. <https://github.com/xinntao/Real-ESRGAN>
2. <https://github.com/TencentARC/GFPGAN>
3. <https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan>
4. <https://github.com/Tencent/ncnn>
5. <https://github.com/Tencent/ncnn/tree/master/tools/pnnx>
6. <https://github.com/pnnx/pnnx>
7. <https://github.com/deepcam-cn/yolov5-face>
8. <https://github.com/derronqi/yolov7-face>
9. <https://github.com/derronqi/yolov8-face>
10. <https://github.com/FeiGeChuanShu/GFPGAN-ncnn>
11. <https://github.com/ultralytics/ultralytics>

## Download Model files (GFPGAN-ESRGAN-ncnn-onnx model files)
https://github.com/CXVUSER/Upscayl-gpfgan-realesr-ncnn-directml/releases/download/v0.0.1-models/models.7z
