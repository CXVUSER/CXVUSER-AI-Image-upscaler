cmake_minimum_required(VERSION 3.12)
project(upsampler_bin)

include(FetchContent)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(TARGET_ARCH ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID})
if(TARGET_ARCH STREQUAL AMD64)
    set(TARGET_ARCH x64)
endif()

set(CMAKE_BUILD_TYPE Release)

# -------------- Support C++17 for using filesystem  ------------------#
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# set(CMAKE_CXX_CLANG_TIDY
#         clang-tidy;
#         -header-filter=.;
#         -checks=*;
#         -warnings-as-errors=*;)

# set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_PROGRAM}")

option(USE_SYSTEM_NCNN "build with system libncnn" OFF)
option(USE_SYSTEM_WEBP "build with system libwebp" OFF)
option(USE_STATIC_MOLTENVK "link moltenvk static library" OFF)

find_package(Threads)
find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)

if (Vulkan_FOUND)
    message(STATUS "Vulkan_INCLUDE_DIRS: ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "Vulkan_LIBRARIES: ${Vulkan_LIBRARIES}")
    message(STATUS "Vulkan_LIBRARY: ${Vulkan_LIBRARY}")
    message(STATUS "Vulkan_FOUND: ${Vulkan_FOUND}")
else ()
    message(FATAL_ERROR "Vulkan Not Found!")
endif ()

find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator PATHS $ENV{VULKAN_SDK}/bin NO_CMAKE_FIND_ROOT_PATH)
message(STATUS "Found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")

macro(compile_shader SHADER_SRC)
    set(SHADER_SRC_FULLPATH ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC})

    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -V -s -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_FILE})

    # fp16 storage
    set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")

    set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -V -s -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_fp16s_SPV_HEX_FILE})

    # int8 storage
    set(SHADER_int8s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_int8s")

    set(SHADER_int8s_SPV_HEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_int8s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_int8s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -DNCNN_int8_storage=1 -V -s -x -o ${SHADER_int8s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_int8s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_int8s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_int8s_SPV_HEX_FILE})
endmacro()

find_package(OpenCV REQUIRED)

if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "OpenCV Not Found!")
endif ()

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "OpenMP Not Found!")
endif ()

if(USE_SYSTEM_NCNN)
    set(GLSLANG_TARGET_DIR "GLSLANG-NOTFOUND" CACHE PATH "Absolute path to glslangTargets.cmake directory")
    if(NOT GLSLANG_TARGET_DIR AND NOT DEFINED ENV{GLSLANG_TARGET_DIR})
        message(WARNING "GLSLANG_TARGET_DIR must be defined! USE_SYSTEM_NCNN will be turned off.")
        set(USE_SYSTEM_NCNN OFF)
    else()
        message(STATUS "Using glslang install located at ${GLSLANG_TARGET_DIR}")

        find_package(Threads)

        include("${GLSLANG_TARGET_DIR}/OSDependentTargets.cmake")
        include("${GLSLANG_TARGET_DIR}/OGLCompilerTargets.cmake")
        if(EXISTS "${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
            # hlsl support can be optional
            include("${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
        endif()
        include("${GLSLANG_TARGET_DIR}/glslangTargets.cmake")
        include("${GLSLANG_TARGET_DIR}/SPIRVTargets.cmake")

        if (NOT TARGET glslang OR NOT TARGET SPIRV)
            message(WARNING "glslang or SPIRV target not found! USE_SYSTEM_NCNN will be turned off.")
            set(USE_SYSTEM_NCNN OFF)
        endif()
    endif()
endif()

if(USE_SYSTEM_NCNN)
    find_package(ncnn)
    if(NOT TARGET ncnn)
        message(WARNING "ncnn target not found! USE_SYSTEM_NCNN will be turned off.")
        set(USE_SYSTEM_NCNN OFF)
    endif()
endif()

if(NOT USE_SYSTEM_NCNN)
    # build ncnn library
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ncnn/CMakeLists.txt")
        message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    option(NCNN_VULKAN "vulkan compute support" ON)
    option(NCNN_THREADS "" ON)
    option(NCNN_OPENMP "" ON)
    option(NCNN_VULKAN_ONLINE_SPIRV "" ON)

    option(NCNN_PIXEL "convert and resize from/to image pixel" ON)
    option(NCNN_PIXEL_ROTATE "rotate image pixel orientation" ON)
    option(NCNN_PIXEL_AFFINE "warp affine image pixel" ON)
    option(NCNN_PIXEL_DRAWING "draw basic figure and text" ON)
    option(NCNN_INT8 "int8 inference" ON)
    option(NCNN_BF16 "bf16 inference" ON)


    option(NCNN_INSTALL_SDK "" OFF)
    option(NCNN_PIXEL_ROTATE "" OFF)
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_DISABLE_RTTI "" OFF)
    option(NCNN_DISABLE_EXCEPTION "" OFF)
    option(NCNN_SYSTEM_GLSLANG "" OFF)
    option(NCNN_SHARED_LIB "" OFF)

    add_subdirectory(external/ncnn)
endif()

if(USE_SYSTEM_WEBP)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
    find_package(WebP)
    if(NOT TARGET webp)
        message(WARNING "webp target not found! USE_SYSTEM_WEBP will be turned off.")
        set(USE_SYSTEM_WEBP OFF)
    endif()
endif()

if(NOT USE_SYSTEM_WEBP)
    # build libwebp library
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/libwebp/CMakeLists.txt")
        message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    option(WEBP_ENABLE_SIMD "" ON)
    option(WEBP_BUILD_ANIM_UTILS "" OFF)
    option(WEBP_BUILD_CWEBP "" OFF)
    option(WEBP_BUILD_DWEBP "" OFF)
    option(WEBP_BUILD_GIF2WEBP "" OFF)
    option(WEBP_BUILD_IMG2WEBP "" OFF)
    option(WEBP_BUILD_VWEBP "" OFF)
    option(WEBP_BUILD_WEBPINFO "" OFF)
    option(WEBP_BUILD_WEBPMUX "" OFF)
    option(WEBP_BUILD_EXTRAS "" OFF)
    option(WEBP_BUILD_WEBP_JS "" OFF)
    option(WEBP_NEAR_LOSSLESS "" OFF)
    option(WEBP_ENABLE_SWAP_16BIT_CSP "" OFF)

    add_subdirectory(external/libwebp)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libwebp/src)
endif()

set(NCNN_GFPGAN_LINK_LIBRARIES ncnn webp ${Vulkan_LIBRARY})

# look for vulkan compute shader and compile
set(SHADER_SPV_HEX_FILES)

compile_shader(realesrgan_preproc.comp)
compile_shader(realesrgan_postproc.comp)
compile_shader(realesrgan_preproc_tta.comp)
compile_shader(realesrgan_postproc_tta.comp)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(WIN32)
# -----------------------------------------------------------------------------
# onnxruntime
# -----------------------------------------------------------------------------
FetchContent_Declare(
    ort
    URL https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.20.1
)

FetchContent_MakeAvailable(ort)

set(ort_bin_dir "${ort_SOURCE_DIR}/runtimes/win-${TARGET_ARCH}/native")

add_library(ort INTERFACE)
target_include_directories(ort INTERFACE "${ort_SOURCE_DIR}/build/native/include")
target_link_libraries(ort INTERFACE "${ort_bin_dir}/onnxruntime.lib")

# -----------------------------------------------------------------------------
# directx-headers - to get the latest DXCore header with GENERIC_ML GUID
# -----------------------------------------------------------------------------
FetchContent_Declare(
    dxheaders
    GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers
    GIT_TAG de28d93dfa9ebf3e473127c1c657e1920a5345ee # v1.613.1
)

FetchContent_MakeAvailable(dxheaders)

set(ort_bin_dir "${ort_SOURCE_DIR}/runtimes/win-${TARGET_ARCH}/native")

add_library(dxheaders INTERFACE)
target_include_directories(dxheaders INTERFACE ${dxheaders_SOURCE_DIR}/include/directx)
target_link_libraries(dxheaders INTERFACE Microsoft::DirectX-Guids)

# -----------------------------------------------------------------------------
# directml
# -----------------------------------------------------------------------------
FetchContent_Declare(
    dml
    URL https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4
)

FetchContent_MakeAvailable(dml)

set(dml_bin_dir "${dml_SOURCE_DIR}/bin/${TARGET_ARCH}-win")

add_library(dml INTERFACE)
target_include_directories(dml INTERFACE "${dml_SOURCE_DIR}/include")
target_link_libraries(dml INTERFACE "${dml_bin_dir}/directml.lib")
elseif(LINUX)
# -----------------------------------------------------------------------------
# onnxruntime
# -----------------------------------------------------------------------------
FetchContent_Declare(
    ort
    URL https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/1.16.3
)

FetchContent_MakeAvailable(ort)

set(ort_bin_dir "${ort_SOURCE_DIR}/runtimes/linux-${TARGET_ARCH}/native")

add_library(ort INTERFACE)
target_include_directories(ort INTERFACE "${ort_SOURCE_DIR}/build/native/include")
target_link_libraries(ort INTERFACE "${ort_bin_dir}/onnxruntime.lib")
elseif(APPLE)
# -----------------------------------------------------------------------------
# onnxruntime
# -----------------------------------------------------------------------------
FetchContent_Declare(
    ort
    URL https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/1.20.1
)

FetchContent_MakeAvailable(ort)

set(ort_bin_dir "${ort_SOURCE_DIR}/runtimes/osx-${TARGET_ARCH}/native")

add_library(ort INTERFACE)
target_include_directories(ort INTERFACE "${ort_SOURCE_DIR}/build/native/include")
target_link_libraries(ort INTERFACE "${ort_bin_dir}/onnxruntime.lib")

endif()

add_executable(${CMAKE_PROJECT_NAME} main.cpp gfpgan.cpp Faceyolov7_lite_e.cpp Faceyolov5bl.cpp retinaface.cpp helpers.cpp generator.cpp encoder.cpp codeformer.cpp pipeline.cpp realesrgan.cpp)
add_dependencies(${CMAKE_PROJECT_NAME} generate-spirv)
if(USE_STATIC_MOLTENVK)
    find_library(CoreFoundation NAMES CoreFoundation)
    find_library(Foundation NAMES Foundation)
    find_library(Metal NAMES Metal)
    find_library(QuartzCore NAMES QuartzCore)
    find_library(CoreGraphics NAMES CoreGraphics)
    find_library(Cocoa NAMES Cocoa)
    find_library(IOKit NAMES IOKit)
    find_library(IOSurface NAMES IOSurface)

    list(APPEND REALESRGAN_LINK_LIBRARIES
        ${Metal}
        ${QuartzCore}
        ${CoreGraphics}
        ${Cocoa}
        ${IOKit}
        ${IOSurface}
        ${Foundation}
        ${CoreFoundation}
    )
endif()

if(WIN32)
target_link_libraries(${CMAKE_PROJECT_NAME} ${NCNN_GFPGAN_LINK_LIBRARIES} ${OpenCV_LIBS} ort dml d3d12 dxcore dxheaders)
elseif(LINUX)
target_link_libraries(${CMAKE_PROJECT_NAME} ${NCNN_GFPGAN_LINK_LIBRARIES} ${OpenCV_LIBS} ort)
else()
target_link_libraries(${CMAKE_PROJECT_NAME} ${NCNN_GFPGAN_LINK_LIBRARIES} ${OpenCV_LIBS} ort)
endif()