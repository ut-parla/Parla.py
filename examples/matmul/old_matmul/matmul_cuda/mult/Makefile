all: gemm

gemm: gemm.o
	g++ -shared -L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart -lcublas gemm.cpp gemm.o -o libgemm.so

gemm.o:
	nvcc -shared -c gemm.cu -Xcompiler -fPIC
