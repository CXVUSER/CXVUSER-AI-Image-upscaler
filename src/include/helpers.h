// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define UNICODE
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NODRAWTEXT
#define NOGDI
#define NOBITMAP
#define NOMCX
#define NOSERVICE
#define NOHELP



#include <optional>
#include <iostream>
#include <filesystem>
#include <span>
#include <string>

#include "half.hpp"
#include "onnxruntime_cxx_api.h"

#if defined(_WIN32)
#include <Windows.h>
#include <DirectML.h>
#include <d3d12.h>
#include <dxcore.h>
#include <wrl/client.h>
//#include <wic_image.h>
#include <wincodec.h>
#include "dml_provider_factory.h"
#endif

enum class ChannelOrder
{
    RGB,
    BGR,
};

bool pathisfolderw(wchar_t *c);
bool pathisfoldera(char *c);
wchar_t *getfilew(wchar_t *t);
char *getfilea(char *t);

#if defined(_WIN32)
std::tuple<Microsoft::WRL::ComPtr<IDMLDevice>, Microsoft::WRL::ComPtr<ID3D12CommandQueue>> CreateDmlDeviceAndCommandQueue(
    std::string_view adapterNameFilter = ""
);
#endif

// Converts a pixel buffer to an NCHW tensor (batch size 1).
// Source: buffer of RGB pixels (HWC) using uint8 components.
// Target: buffer of RGB planes (CHW) using float32/float16 components.
template <typename T> 
void CopyPixelsToTensor(
    std::span<const std::byte> src, 
    std::span<std::byte> dst,
    uint32_t height,
    uint32_t width,
    uint32_t channels
);

// Converts an NCHW tensor buffer (batch size 1) to a pixel buffer.
// Source: buffer of RGB planes (CHW) using float32/float16 components.
// Target: buffer of RGB pixels (HWC) using uint8 components.
template <typename T>
void CopyTensorToPixels(
    std::span<const std::byte> src, 
    std::span<BYTE> dst,
    uint32_t height,
    uint32_t width,
    uint32_t channels
);

#if defined(_WIN32)
void FillNCHWBufferFromImageFilename(
    std::wstring_view filename,
    std::span<std::byte> tensorBuffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    ONNXTensorElementDataType bufferDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ChannelOrder bufferChannelOrder = ChannelOrder::RGB
);

void SaveNCHWBufferToImageFilename(
    std::wstring_view filename,
    std::span<const std::byte> tensorBuffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    ONNXTensorElementDataType bufferDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ChannelOrder bufferChannelOrder = ChannelOrder::RGB
);
#endif