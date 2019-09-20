project(cholesky_bench)
cmake_minimum_required(VERSION 3.15)

list(PREPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

include(UseCython)

find_package(CUDA REQUIRED)
find_package(Magma REQUIRED)
find_package(Python COMPONENTS Development)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

cython_add_module(cholesky_bench cholesky_bench_pyx True)

target_include_directories(cholesky_bench PRIVATE ${CUDA_INCLUDE_DIRS} ${Magma_INCLUDE_DIR})

target_link_directories(cholesky_bench PRIVATE ${Magma_LIB_DIR})

target_link_libraries(cholesky_bench PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${Magma_LIBRARIES})

install(TARGETS cholesky_bench LIBRARY DESTINATION cholesky_bench)
