#include<stdio.h>
#include<string>
#include<iostream>
#include<iomanip>
#include<cuda.h>
#include<cuda_runtime_api.h>
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
	long n_local = strtol(argv[2], NULL, 10);
	long trials = strtol(argv[3], NULL, 10);

	std::cout << "NGPUS: " << m << std::endl;
	std::cout << "n_local: "<< n_local << std::endl;

	long N = n_local * m;

	//Allocate and initialize host array
	double* test_vector = (double*) malloc(N*sizeof(double));
	#pragma omp parallel for
	for(int i=0; i<N; i++){
		test_vector[i] = i;
	}

	omp_set_num_threads(m);
	for(int k = 0; k < trials; ++k){
	auto t_e2e_start = high_resolution_clock::now();
	#pragma omp parallel
	{
		unsigned int thread_id = omp_get_thread_num();
		unsigned int num_threads = omp_get_num_threads();

		std::cout << "Thread: " << thread_id << std::endl;
		cudaSetDevice(thread_id);

		double* device_array = 0;
		cudaMalloc((void**)&device_array, n_local*sizeof(double));
		cudaMemset(device_array, 0, n_local);
		#pragma omp barrier
		auto t_copy_start = high_resolution_clock::now();
		cudaMemcpy(device_array, test_vector+thread_id*n_local, sizeof(double)*n_local, cudaMemcpyHostToDevice);
		//cudaDeviceSynchronize();
		auto t_copy_stop = high_resolution_clock::now();
		
		#pragma omp barrier
		cudaFree(device_array);

		auto copy_time = duration_cast<duration<double>>(t_copy_stop - t_copy_start);

		std::cout << "Thread: " << thread_id
		     	  << ":: Copy time: "<<  copy_time.count() << std::endl;

	}
	auto t_e2e_stop = high_resolution_clock::now();

	auto e2e_time = duration_cast<duration<double>>(t_e2e_stop - t_e2e_start); 
	std::cout << " e2e time: "<<  e2e_time.count() << std::endl;
	}
	free(test_vector);
	return 0;
}
