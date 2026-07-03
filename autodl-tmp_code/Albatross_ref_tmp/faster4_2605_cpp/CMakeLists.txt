cmake_minimum_required(VERSION 3.24)
project(rwkv7_fast_v4 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(RWKV7_FAST_V4_SOURCES
  src/rwkv7_fast_v4.cu
  src/pth_archive.cpp
  src/pth_tensor.cpp
  cuda/rwkv7_v3a_ops.cu
  cuda/rwkv7_fast_ops_fp16.cu
  cuda/rwkv7_wkv_fp16_v2.cu
  cuda/rwkv7_wkv_fp32_v2.cu
)

add_executable(rwkv7_fast_v4
  ${RWKV7_FAST_V4_SOURCES}
)

target_include_directories(rwkv7_fast_v4 PRIVATE cuda include)
target_link_libraries(rwkv7_fast_v4 PRIVATE CUDA::cublas CUDA::cublasLt)
target_compile_definitions(rwkv7_fast_v4 PRIVATE _IO_FP16_)
set_property(TARGET rwkv7_fast_v4 PROPERTY CUDA_RUNTIME_LIBRARY Shared)

if(MSVC)
  target_compile_definitions(rwkv7_fast_v4 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  target_compile_options(rwkv7_fast_v4 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /utf-8 /EHsc>
  )
else()
  target_compile_options(rwkv7_fast_v4 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endif()

target_compile_options(rwkv7_fast_v4 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extra-device-vectorization --ptxas-options=-O3>
)
