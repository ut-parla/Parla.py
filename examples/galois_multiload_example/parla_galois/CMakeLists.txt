cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(parla_galois CXX)

include_directories(/home/amp/LocalInstalls/galois/include/ /home/amp/LocalInstalls/conda/miniconda3/envs/parla/include)

add_library(bfs_cpp SHARED bfs_cpp.cpp)