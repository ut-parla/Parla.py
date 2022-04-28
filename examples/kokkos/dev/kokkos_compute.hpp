#include<Kokkos_Core.hpp>
#include<stdio.h>
#include<string>
#include<iostream>
#include<iomanip>
#include<impl/Kokkos_Timer.hpp>

#ifdef ENABLE_CUDA
	#include<cuda_runtime.h>
	#include<Kokkos_Cuda.hpp>
	//#include<helper_cuda.h>
#endif


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


//Generic function that maps 1D array to double
//Here we implement a reduction-sum
double kokkos_function(double* array, const int N, const int dev_id){
	
	std::cout<< "Running on Device" << dev_id <<std::endl;
	
	//Create cuda stream for current device
	//This enitre ifdef step is now unnecessary, but is a good safety
	//The kernel will not launch if cudaSetDevice does not match the device set in kokkos's internal singleton
	#ifdef ENABLE_CUDA
		cudaSetDevice(dev_id);
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		Kokkos::Cuda cuda1(stream);
		auto range_policy = Kokkos::RangePolicy<Kokkos::Cuda>(cuda1, 0, N);
	#else
		auto range_policy = Kokkos::RangePolicy<Kokkos::Threads>(0, N);
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

