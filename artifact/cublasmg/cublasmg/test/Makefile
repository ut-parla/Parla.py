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

CUDAMG_ROOT ?= /work2/06081/wlruys/frontera/cublasmg0.1.0-cuda10.1-ubuntu16.04-x86_64/cudalibmg
CUBLASMG_ROOT ?= /work2/06081/wlruys/frontera/cublasmg0.1.0-cuda10.1-ubuntu16.04-x86_64/cublasmg
CUDA_ROOT ?= ${TACC_CUDA_DIR}

CXX=g++
CXXFLAGS=-std=c++11 -march=native -O0

CUDA_INC=-I${CUDA_ROOT}/include -I${CUBLASMG_ROOT}/include -I${CUDAMG_ROOT}/include
CUDA_LIB=-L${CUBLASMG_ROOT}/lib -lcublasMg -L${CUDAMG_ROOT}/lib -lcudalibmg
CUDA_LIB+=-L${CUDA_ROOT}/lib64 -lcudart -lcublas -lnvToolsExt

all:
	${CXX} 1gpuGEMM.cpp ${CUDA_INC} ${CUDA_LIB} -o 1gpuGEMM.exe ${CXXFLAGS}
	${CXX} 2gpuGEMM.cpp ${CUDA_INC} ${CUDA_LIB} -o 2gpuGEMM.exe ${CXXFLAGS}
	${CXX} 4gpuGEMM.cpp ${CUDA_INC} ${CUDA_LIB} -o 4gpuGEMM.exe ${CXXFLAGS}

clean:
	rm -rf *.exe
