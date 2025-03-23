cmake_minimum_required(VERSION 3.22)
project(GPUSolver)

set(CMAKE_CXX_STANDARD 17)
set(CPPVERSION 17)
set(SM_ARCH 86)

set(CMAKE_CUDA_SEPARABLE_COMPILATION  ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES ${SM_ARCH})
set(CMAKE_CUDA_STANDARD ${CPPVERSION})
set(CMAKE_CXX_STANDARD ${CPPVERSION})
set(CMAKE_CXX_FLAGS "-std=c++${CPPVERSION}")
set(CMAKE_CUDA_FLAGS "-std=c++${CPPVERSION}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "-std=c++${CPPVERSION}")

find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)
add_executable(GPU_runner)
add_executable(CPU_runner)
add_executable(CGSolver)

# SOURCE
# ------------------------------------------
target_sources(CPU_runner
        PRIVATE
        cpu/bse_solvers.cpp
        cpu/bse_solvers.h
)
target_sources(GPU_runner
        PRIVATE
        gpu/bse_solvers_parallel.cuh
        main.cu
)
target_sources(CGSolver
        PRIVATE
        gpu/cg_solver.cu
)

# PROPERTIES
# ------------------------------------------
set_target_properties(GPU_runner
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(CPU_runner
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(CGSolver
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)


# LINK LIBRARIES
# ------------------------------------------
target_link_libraries(CGSolver PRIVATE
        CUDA::cusparse
        CUDA::cublas
        CUDA::cudart
)

target_link_libraries(GPU_runner PRIVATE
        CUDA::cusparse
        CUDA::cudart
)

