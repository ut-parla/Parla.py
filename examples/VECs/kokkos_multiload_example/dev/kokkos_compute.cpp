#include<Kokkos_Core.hpp>
#include<stdio.h>
#include<string>
#include<iostream>
#include<iomanip>
#include<impl/Kokkos_Timer.hpp>
#include"kokkos_compute.hpp"

typedef Kokkos::DefaultExecutionSpace DeviceSpace;

//C++ script to test Kokkos function
int main(int argc, char* argv[]){
	int N = 1000;
	Kokkos::ScopeGuard kokkos_scope(argc, argv);
	double* test_vector = (double*) Kokkos::kokkos_malloc<DeviceSpace>(N*sizeof(double));
	
	Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> array_view(test_vector, N);
	
	//Initialize vector to 1..N
	Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i){
		array_view(i) = 1;
	});

	//Sum with reduction
	double result = kokkos_function(test_vector, N);
	std::cout << result << std::endl;
	//free(test_vector);
	return 0;
}
