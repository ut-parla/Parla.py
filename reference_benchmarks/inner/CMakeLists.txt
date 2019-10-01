project(inner_reference_benchmark)
cmake_minimum_required(VERSION 3.8)

FIND_PACKAGE(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(inner inner.cpp)

target_include_directories(inner PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(inner ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
