cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(Paean LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -fuse-ld=gold -fstack-protector")

set(CUDA_MIN_VERSION "9.0")
find_package(CUDA ${CUDA_MIN_VERSION})
if (NOT CUDA_FOUND)
	message(FATAL_ERROR "CUDA >= ${CUDA_MIN_VERSION} Required, Found ${CUDA_VERSION}.")
endif()

# check CUDA version
set(GPU_ARCHS 53 61 70)
if (CUDA_VERSION VERSION_GREATER_EQUAL 10.0)
	list(APPEND GPU_ARCHS 75)
endif()
if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
	list(APPEND GPU_ARCHS 80)
    add_definitions(-DTHRUST_IGNORE_CUB_VERSION_CHECK)
    add_definitions(-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT)
endif()
foreach(arch ${GPU_ARCHS})
	set(CUDA_GENCODE "${CUDA_GENCODE} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
set(CMAKE_CUDA_STANDARD 11)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_LIBRARY_DIR /usr/local/cuda/lib64/)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(Thrust)
find_package(HTSlib)
find_package(Threads)
find_package(ZLIB REQUIRED)

set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/third_party)

# fmt
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/fmt)
# gff
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/gff)

# for parse
add_library(parse STATIC
    ${CMAKE_SOURCE_DIR}/src/parse.cpp)
target_include_directories(parse PRIVATE
    ${HTSlib_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIRS})
target_link_libraries(parse
    ${HTSlib_LIBRARY} ${ZLIB_LIBRARIES}
    gff Threads::Threads fmt::fmt)

# for cub sort
add_library(cubsort STATIC
    ${CMAKE_SOURCE_DIR}/src/cub_sort.cu)
target_include_directories(cubsort PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${THRUST_INCLUDE_DIR})

# for features like bin length and fusion
add_library(feature STATIC
    ${CMAKE_SOURCE_DIR}/src/fusion.cpp)

# for paean
add_executable(paean ${CMAKE_SOURCE_DIR}/src/gene.cu)
target_include_directories(paean PRIVATE
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(paean parse feature cubsort)

