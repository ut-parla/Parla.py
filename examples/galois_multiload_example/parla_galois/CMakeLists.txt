cmake_minimum_required(VERSION 3.13)
project(parla_galois CXX)

include_directories(/home/amp/LocalInstalls/galois/include/ /home/amp/LocalInstalls/conda/miniconda3/envs/parla/include)

add_library(bfs_cpp SHARED bfs_cpp.cpp)