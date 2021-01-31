cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(Paean LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -fstack-protector")

if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "sm_61" CACHE STRING "Target GPU architecture")
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=${CUDA_ARCH} -use_fast_math)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_LIBRARY_DIR /usr/local/cuda/lib64/)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(Thrust)
find_package(HTSlib)
find_package(Threads)
find_package(ZLIB REQUIRED)

# fmt
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/fmt)

set(PAEAN_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(PAEAN_INCLUDE_DIR ${PAEAN_SOURCE_DIR}/include)
set(GFF_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/gff)
set(CUB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cub)
set(HASHMAP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/robin_hood_map)

file(GLOB GFF_SRCS ${GFF_SOURCE_DIR}/src/*.cpp)
file(GLOB PAEAN_CU_SRCS ${PAEAN_SOURCE_DIR}/src/gene.cu)

# for parse
add_library(parse STATIC
    ${GFF_SRCS}
    ${PAEAN_SOURCE_DIR}/src/parse.cpp)
target_include_directories(parse PRIVATE
    ${PAEAN_INCLUDE_DIR}
    ${HTSlib_INCLUDE_DIR}
    ${GFF_SOURCE_DIR}/include
    ${HASHMAP_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIRS})
target_link_libraries(parse
    ${HTSlib_LIBRARY} ${ZLIB_LIBRARIES}
    Threads::Threads fmt::fmt)

# for cub sort
add_library(cubsort STATIC
    ${PAEAN_SOURCE_DIR}/include/cub_sort.cuh
    ${PAEAN_SOURCE_DIR}/src/cub_sort.cu)
target_include_directories(cubsort PRIVATE
    ${PAEAN_INCLUDE_DIR}
    ${CUB_INCLUDE_DIR}
    ${THRUST_INCLUDE_DIR})
set_property(TARGET cubsort
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# for features like bin length and fusion
add_library(feature STATIC
    ${PAEAN_SOURCE_DIR}/src/fusion.cpp)
target_include_directories(feature PRIVATE
    ${PAEAN_INCLUDE_DIR})

# for paean
add_executable(paean ${PAEAN_CU_SRCS})
target_include_directories(paean PRIVATE
    ${PAEAN_INCLUDE_DIR})
set_property(TARGET paean
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(paean parse feature cubsort)

