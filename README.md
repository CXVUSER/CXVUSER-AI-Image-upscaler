# CXVUSER-AI-Image-upscaler 🚀

My Application for Restoration (low-res, old photo) and colorization images with AI

This project uses (onnx, ncnn inference) with Vulkan and DirectML compute GPU acceleration...

Platforms:
1. Windows (current)
2. Linux (at future)
3. MacOS (at future)

## :construction: AI Models support :construction:

1. GFPGANCleanv1-NoCE-C2 (NCNN)
2. GFPGAN 1.2,1.3,1.4 (ONNX)
3. ncnn ESRGAN models
4. CodeFormer 0.1.0
5. RestoreFormer, RestoreFormer-plus-plus
6. GPEN
7. Siggraph17, DDColor, Deoldify for colorize photo

### Compile:
## Clone Project and Get Submodules

Make sure submodules are initialized and updated

```console
git clone https://github.com/CXVUSER/CXVUSER-AI-Image-upscaler.git
git submodule update --init --recursive
```

## Clone project with Submodules

```sh
git clone --recursive https://github.com/CXVUSER/CXVUSER-AI-Image-upscaler.git
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