cmake_minimum_required(VERSION 3.15)
project(parla_runtime_libs C)

set(CMAKE_C_STANDARD 11)

add_library(parla_supervisor SHARED virt_dlopen.c supervisor_wrappers.c)
target_link_libraries(parla_supervisor dl)

add_library(parla_context SHARED affinity_override.c)
target_link_libraries(parla_context dl)
