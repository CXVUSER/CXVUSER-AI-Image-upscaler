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

### Usage tips:
```
Usage: this_binary [options]...

 -i <img>      Path to input image
 -s <digit>    Model scale factor (default=autodetect)
 -j <digit>    Custom output scale factor
 -t <digit>    Tile size (default=auto)
 -f            Restore faces (default=CodeFormer)
 -m <string>   ESRGAN model name (default=./models/ESRGAN/4xNomos8kSC)
 -g <string>   GFPGAN model path (default=./models/face_restore/codeformer_0_1_0.onnx)
 -x <digit>    Face detection threshold (default=0.5, recommended range: 0.3-0.7)
 -c            Use CodeFormer face restore model (ncnn)
 -d            Switch face restore inference to ONNX (default=enabled)
 -w <digit>    CodeFormer Fidelity (Only ONNX, default=0.7)
 -u            Face upsample (after face restore)
 -z <string>   FaceUpsample model (ESRGAN)
 -p            Use face parsing for accurate face masking (default=false)
 -o <string>   Override image input path
 -l <string>   Face detector model (default=y7, options: y7, y5, (RetinaFace: rt, mnet))
 -h            Colorize grayscale photo with DeOldify Artistic
 -n            No upsample
 -a            Wait (pause execution)
 -v            Verbose mode (detailed logging)
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