cmake_minimum_required(VERSION 3.8)
if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()
project(foundationpose_cpp)

option(ENABLE_TENSORRT "Build TensorRT-backed inference targets." ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(ENABLE_TENSORRT)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)

  file(GLOB TENSORRT_CUDA_ROOTS "/usr/local/cuda-*")
  set(TENSORRT_ROOT_CANDIDATES
      $ENV{TENSORRT_ROOT}
      $ENV{CONDA_PREFIX}
      ${CUDA_TOOLKIT_ROOT_DIR}
      /usr
      /usr/local/cuda
      ${TENSORRT_CUDA_ROOTS})

  find_path(TENSORRT_INCLUDE_DIR
            NvInfer.h
            PATHS ${TENSORRT_ROOT_CANDIDATES}
            PATH_SUFFIXES include include/x86_64-linux-gnu targets/x86_64-linux/include)
  find_library(TENSORRT_NVINFER_LIBRARY
               nvinfer
               PATHS ${TENSORRT_ROOT_CANDIDATES}
               PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu targets/x86_64-linux/lib)
  find_library(TENSORRT_NVONNXPARSER_LIBRARY
               nvonnxparser
               PATHS ${TENSORRT_ROOT_CANDIDATES}
               PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu targets/x86_64-linux/lib)
  find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY
               nvinfer_plugin
               PATHS ${TENSORRT_ROOT_CANDIDATES}
               PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu targets/x86_64-linux/lib)

  if(NOT TENSORRT_INCLUDE_DIR OR
     NOT TENSORRT_NVINFER_LIBRARY OR
     NOT TENSORRT_NVONNXPARSER_LIBRARY OR
     NOT TENSORRT_NVINFER_PLUGIN_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please install it or export your TENSORRT_ROOT.")
  endif()

  get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_NVINFER_LIBRARY}" DIRECTORY)
endif()

add_subdirectory(easy_deploy_tool/deploy_core)
if(ENABLE_TENSORRT)
  add_subdirectory(easy_deploy_tool/inference_core/trt_core)
endif()
add_subdirectory(detection_6d_foundationpose)

add_subdirectory(simple_tests)

if(ENABLE_TENSORRT)
  if(TARGET trt_core)
    target_compile_options(trt_core PRIVATE "SHELL:-idirafter ${TENSORRT_INCLUDE_DIR}")
    set_property(TARGET trt_core PROPERTY LINK_LIBRARIES
      deploy_core
      CUDA::cudart
      ${TENSORRT_NVINFER_LIBRARY}
      ${TENSORRT_NVONNXPARSER_LIBRARY}
      ${TENSORRT_NVINFER_PLUGIN_LIBRARY}
    )
    set_property(TARGET trt_core PROPERTY INTERFACE_LINK_LIBRARIES deploy_core)
  endif()

  add_executable(onnx_to_trt tools/onnx_to_trt.cpp)
  target_compile_features(onnx_to_trt PRIVATE cxx_std_17)
  target_compile_options(onnx_to_trt PRIVATE "SHELL:-idirafter ${TENSORRT_INCLUDE_DIR}")
  target_link_libraries(onnx_to_trt PRIVATE
    CUDA::cudart
    ${TENSORRT_NVINFER_LIBRARY}
    ${TENSORRT_NVONNXPARSER_LIBRARY}
    ${TENSORRT_NVINFER_PLUGIN_LIBRARY}
  )
endif()
