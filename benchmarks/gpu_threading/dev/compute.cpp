//#include<Kokkos_Core.hpp>
#include<stdio.h>
#include<string>
#include<iostream>
#include<iomanip>
#include<cuda.h>
#include<cuda_runtime_api.h>
//#include<impl/Kokkos_Timer.hpp>
#include<stdlib.h>
#include<chrono>
#include<ctime>
#include<cmath>
#include<omp.h>

double kokkos_function_copy(double* d_array, double* array, const int N, const int dev_id){

	cudaSetDevice(dev_id);
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	std::cout << "Running on Device " << dev_id << std::endl;

	cudaMemcpyAsync(d_array, array, sizeof(double)*N, cudaMemcpyHostToDevice, stream);
	cudaDeviceSynchronize();
	//cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return N;
};

using namespace std::chrono;

int main(int argc, char* argv[]){

	long m = strtol(argv[1], NULL, 10);

	std::cout << "NGPUS: "<< m <<std::endl;
	long n_local = 1000000000; 
	long N = n_local * m;

	double* test_vector = (double*) malloc(N*sizeof(double));
	
	std::cout<<"Starting"<< test_vector << std::endl;
	#pragma omp parallel for
	for(int i=0; i<N; i++){
		test_vector[i] = i;
	}

	omp_set_num_threads(m);

	#pragma omp parallel
	{

	std::cout<<"Starting "<< omp_get_num_threads()<<std::endl;

	#pragma omp for
	for(int i = 0; i < m; ++i){
		double result = 0.0;
		double* device_array;
		cudaSetDevice(i);
		cudaMalloc((void**)&device_array, sizeof(double)*n_local);

		for(int k=0; k<1; ++k){
			//Sum with reduction
			result = kokkos_function_copy(device_array,test_vector, n_local, i);
			std::cout<< omp_get_thread_num() << "Result :: " << result <<std::endl;
		}

		cudaFree(device_array);
	}

	}

	int max_iter = 2;
	#pragma omp parallel
	{

	std::cout<<"Starting "<< omp_get_num_threads()<<std::endl;

	#pragma omp for
	for(int i = 0; i < m; ++i){

		double* device_array;
		cudaSetDevice(i);
		cudaMalloc((void**)&device_array, sizeof(double)*n_local);
		double result = 0.0;
		for(int k=0; k<max_iter; ++k){
			//Sum with reduction

			auto t1 = high_resolution_clock::now();
			result = kokkos_function_copy(device_array, test_vector+i*n_local, n_local, i);
			std::cout<< omp_get_thread_num() << "Result :: " << result <<std::endl;
			auto t2 = high_resolution_clock::now();

			duration<double> time_span = duration_cast<duration<double>>(t2-t1);

			std::cout << " Time :: " << time_span.count()/max_iter <<std::endl;
		}
		cudaFree(device_array);
	}

	}

	free(test_vector);
	return 0;
}
