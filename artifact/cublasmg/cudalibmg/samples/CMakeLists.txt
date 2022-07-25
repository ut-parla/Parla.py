# 
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

# ---[ Check cmake version.
CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project specIFication.
PROJECT(cudamglib_example C CXX CUDA)

# ---[ Use the default installation path if not set.
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# ---[ Build type
IF(NOT CMAKE_BUILD_TYPE) 
    SET(CMAKE_BUILD_TYPE Develop)
ENDIF(NOT CMAKE_BUILD_TYPE)

# ---[ Specialize for cross-compilation
IF(CMAKE_CROSSCOMPILING)
    MESSAGE(STATUS "Cross compiling: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
    SET(CUDA_TARGET_LIBRARIES "/externals/cuda-10.1-${CMAKE_SYSTEM_PROCESSOR}/lib")
    SET(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/externals/cuda-10.1-${CMAKE_SYSTEM_PROCESSOR}/include")
    SET(CMAKE_CUDA_LINK_EXECUTABLE "<CMAKE_CUDA_HOST_LINK_LAUNCHER> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -L${CUDA_TARGET_LIBRARIES} -lcudadevrt -lcudart_static -lrt -lpthread -ldl")
    SET(CMAKE_CUDA_CREATE_SHARED_LIBRARY  "<CMAKE_CUDA_HOST_LINK_LAUNCHER> <CMAKE_SHARED_LIBRARY_CUDA_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CUDA_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> -L${CUDA_TARGET_LIBRARIES} -lcudadevrt -lcudart_static -lrt -lpthread -ldl")
ELSE()
    MESSAGE(STATUS "Compiling: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
ENDIF()

IF(NOT TARGET cudalibmg)
    LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
    FIND_PACKAGE(cudalibmg REQUIRED)
ENDIF()

# ---[ Example: matrixDescriptor
ADD_SUBDIRECTORY(matrixDescriptor)
# ---[ Example: redistribute
ADD_SUBDIRECTORY(redistribute)
