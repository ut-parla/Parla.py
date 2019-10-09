#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
  std::size_t n = 10000000ull;
  std::size_t partitions = 100ull;
  std::mt19937 gen{0};
  std::uniform_real_distribution<> dis(-1., 1.);
  auto x = std::make_unique<double[]>(n);
  auto y = std::make_unique<double[]>(n);
  for (std::size_t i = 0; i < n; i++) {
    x.get()[i] = dis(gen);
  }
  for (std::size_t i = 0; i < n; i++) {
    y.get()[i] = dis(gen);
  }
  int devices = 0;
  auto stat = cudaGetDeviceCount(&devices);
  assert(stat == cudaSuccess);
  assert(devices > 0);
  auto start = std::chrono::high_resolution_clock::now();
  auto locations = std::make_unique<int[]>(partitions);
  auto x_input_ptrs = std::make_unique<double*[]>(partitions);
  auto y_input_ptrs = std::make_unique<double*[]>(partitions);
  auto sizes = std::make_unique<std::size_t[]>(partitions);
  auto per_device_block_indices = std::make_unique<std::size_t[]>(devices + 1);
  per_device_block_indices.get()[0] = 0;
  int previous_device = 0;
  for (std::size_t i = 0; i < partitions; i++) {
    x_input_ptrs.get()[i] = x.get() + i * (n / partitions);
    y_input_ptrs.get()[i] = y.get() + i * (n / partitions);
    std::size_t default_size = (n + partitions - 1) / partitions;
    std::size_t size = i < partitions - 1 ? default_size : ((n - 1) % default_size) + 1;
    sizes.get()[i] = size;
    int location = (i * devices) / partitions;
    locations.get()[i] = location;
    if (location != previous_device) {
      previous_device = location;
      per_device_block_indices.get()[location] = i;
    }
  }
  per_device_block_indices.get()[devices] = partitions;
  auto partial_sums = std::make_unique<double[]>(partitions);
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < devices; i++) {
    threads.emplace_back([&](std::size_t i) {
      auto stat2 = cudaSetDevice(i);
      assert(stat2 == cudaSuccess);
      cublasHandle_t handle;
      auto stat3 = cublasCreate(&handle);
      assert(stat3 == CUBLAS_STATUS_SUCCESS);
      // Could use streams here, but the point
      // of this benchmark is to show that our orchestration
      // layer doesn't introduce large overheads.
      // Numba exposes streams in Python too, so
      // we could potentially use them in Python too.
      // That would over-complicate the demo though.
      for (std::size_t j = per_device_block_indices.get()[i]; j < per_device_block_indices.get()[i+1]; j++) {
        auto size = sizes.get()[j];
        double *x_dev_block;
        double *y_dev_block;
        stat2 = cudaMalloc(&x_dev_block, size * sizeof(double));
        assert(stat2 == cudaSuccess);
        stat2 = cudaMalloc(&y_dev_block, size * sizeof(double));
        assert(stat2 == cudaSuccess);
        cublasSetVector(size, sizeof(double), x_input_ptrs.get()[j], 1, x_dev_block, 1);
        cublasSetVector(size, sizeof(double), y_input_ptrs.get()[j], 1, y_dev_block, 1);
        cublasDdot(handle, size, x_dev_block, 1, y_dev_block, 1, &partial_sums.get()[j]);
        cudaFree(x_dev_block);
        cudaFree(y_dev_block);
      }
    }, i);
  }
  for (auto &thread : threads) {
    thread.join();
  }
  double result = 0;
  for (std::size_t i = 0; i < partitions; i++) {
    result += partial_sums.get()[i];
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double, std::chrono::seconds::period>(end - start).count() << std::endl;
}
