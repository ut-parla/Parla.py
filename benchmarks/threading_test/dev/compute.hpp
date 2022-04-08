#include<Kokkos_Core.hpp>
#include<stdio.h>
#include<string>
#include<iostream>
#include<iomanip>
#include<impl/Kokkos_Timer.hpp>
#include<chrono>
#include<cuda.h>
#include<cuda_runtime_api.h>

#include <cstdint>
#ifdef ENABLE_CUDA
	#include<cuda_runtime.h>
	#include<Kokkos_Cuda.hpp>
	//#include<helper_cuda.h>
#endif

using namespace std::chrono;
typedef Kokkos::DefaultExecutionSpace DeviceSpace;

void init(int dev){
	Kokkos::InitArguments args;
	args.num_threads=0;
	args.num_numa=0;
	args.device_id=0;
	
	#ifdef ENABLE_CUDA
	args.device_id=dev;
	#endif

	Kokkos::initialize(args);
}

void finalize(){
	Kokkos::finalize();
}

#ifdef ENABLE_CUDA
/*The CUDA kernel */
__global__ void vector_add_cu(float *out, float *a, float *b, int n){
	for(int i = 0; i < n; i++){
		out[i] = a[i] + b[i];
	}
}

__global__ void vector_add_cu(double *out, double *a, double *b, int n){
	for(int i = 0; i < n; i++){
		out[i] = a[i] + b[i];
	}
}

/* Implementation of the function to be wrapped by Cython */
void addition(float *out, float *a, float *b, int N){
    
    float *d_a, *d_b, *d_out;    

    cudaMalloc((void**)&d_a, sizeof(float)*N);
    cudaMalloc((void**)&d_b, sizeof(float)*N);
    cudaMalloc((void**)&d_out, sizeof(float)*N);

    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);

    vector_add_cu<<<1, 1>>>(d_out, d_a, d_b, N);

    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}
#else
void addition(float *out, float *a, float *b, int N){
    for(int i = 0; i < N; ++i){
        out[i] = a[i] + b[i];
    }    
}
#endif



/*
double kokkos_function_copy(double* array, const int N, const int dev_id){

	cudaSetDevice(dev_id);
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	Kokkos::Cuda cuda1(stream);
	auto range_policy = Kokkos::RangePolicy<Kokkos::Cuda>(cuda1, 0, N);

	double* device_array;
	cudaMalloc((void**)&device_array, sizeof(double)*N*N);
	cudaMemcpyAsync(device_array, array, sizeof(double)*N*N, cudaMemcpyHostToDevice, stream);

	int nrepeat = 100;
	
	for(int repeat = 0; repeat < nrepeat; repeat++){

		double result = 0;
		Kokkos::parallel_reduce("xAx", range_policy, KOKKOS_LAMBDA(int j, double &update)		{
				update += (double) j;

		}, result);
		Kokkos::fence();

	}
	
	std::cout << "Running on Device " << dev_id << std::endl;
	cudaStreamDestroy(stream);
	cudaFree(device_array);

	return 1.0;
}
*/

double* copy2dev(double* array, const int N, const int dev_id){

		cudaSetDevice(dev_id);

		std::cout << "Running on Device " << dev_id <<" " << N << " " << std::endl;
		double* device_array;
		cudaMalloc((void**)&device_array, sizeof(double)*N*N);
		cudaMemcpy(device_array, array, sizeof(double)*N*N, cudaMemcpyHostToDevice);

		std::cout << "Finished " << dev_id <<" " << N << " " << std::endl;
		return device_array;
}


void cleanup(double* array, const int dev_id){
	cudaSetDevice(dev_id);
	cudaFree(array);
}

double kokkos_function_copy(double* array, const int N, const int dev_id){

	//Setting range policy and doing explicit copies isn't necessary. The above should work, but this is a safety
		cudaSetDevice(dev_id);
		//cudaStream_t stream;
		//cudaStreamCreate(&stream);
		//Kokkos::Cuda cuda1(stream);
		auto range_policy = Kokkos::RangePolicy<Kokkos::Cuda>(0, N);

		std::cout << "Running on Device Mat " << dev_id << std::endl;

		//double* device_array;
		//cudaMalloc((void**)&device_array, sizeof(double)*N*N);
		//cudaMemcpy(device_array, array, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	int repeat = 3000;
	double sum = 0.0;
	for(int k=0; k < repeat; ++k){
			Kokkos::parallel_reduce("Reduction", range_policy, KOKKOS_LAMBDA(const int i, double& lsum){

				double temp2 = 0.0;
				for(int j = 0; j<N; ++j){
					temp2 += array[i*N + j] * 2;
				}

				lsum += array[N*k+i]*temp2;
			}, sum);
	}
	Kokkos::fence();


	//cudaStreamDestroy(stream);

	return sum;
};

void copy_cupy(double* d_array, double* h_array, const int N, const int dev_id){
    cudaSetDevice(dev_id);

    auto t_copy_start = high_resolution_clock::now();
    //cudaMemcpyAsync(device_array, array, sizeof(double)*N, cudaMemcpyHostToDevice, stream);
    cudaMemcpy(d_array, h_array, sizeof(double)*N, cudaMemcpyHostToDevice);
    auto t_copy_stop = high_resolution_clock::now();

    auto copy_time = duration_cast<duration<double>>(t_copy_stop - t_copy_start);
    std::cout << "DEV: " << dev_id << ":: Copy time: "<<  copy_time.count() << std::endl;
}

//Generic function that maps 1D array to double
//Here we implement a reduction-sum
double kokkos_function(double* array, const int N, const int dev_id){
	
	std::cout<< "Running on Device" << dev_id <<std::endl;
	
	#ifdef ENABLE_CUDA
		//const cudaInternalDevices &dev_info = CudaInternalDevices::singleton();
		//auto cuda_space = Kokkos::Cuda();

	#endif

	//Create cuda stream for current device
	//All ifdefs here are unnecessary, but is a good safety:
	//The kernel will not launch if cudaSetDevice does not match the device set in kokkos's internal singleton
	#ifdef ENABLE_CUDA
		cudaSetDevice(dev_id);
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		Kokkos::Cuda cuda1(stream);
		auto range_policy = Kokkos::RangePolicy<Kokkos::Cuda>(cuda1, 0, N);
	#else
		auto range_policy = Kokkos::RangePolicy<Kokkos::Serial>(0, N);
	#endif

	double sum = 0.0;
	{
	//Kokkos::Timer timer;	
	
	//Turn array into Unmanaged Kokkos View in Default Exec Space
	Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> array_view(array, N);

	//Launch Kokkos Kernel (Perform reduction)
	//Note that the range policy could be left as default and just specify N. 
	Kokkos::parallel_reduce(range_policy, KOKKOS_LAMBDA(const int i, double& lsum){
		//lsum += array[i]; //This also works (but is less robust for debugging)
		lsum += array_view(i);
	}, sum);

	Kokkos::fence();
	}
	//std::cout << "Finished on Device " << dev_id << std::endl;

	#ifdef ENABLE_CUDA
		cudaStreamDestroy(stream);
	#endif
	return sum;
};

