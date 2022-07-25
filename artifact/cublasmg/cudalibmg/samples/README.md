## cudaLibMg Samples (cudaLibMg) 

### README

This is a demo example for using 2D device grid and matrix descriptor.

### REQUIREMENTS

1. Linux (x86_64 or ppc64le)
2. CMAKE-3.10 or later
3. C++11
4. cudaLibMg binaries and headers
4. CUDA Toolkit of a version greater than the one used to compile the cudaLibMg binaries

### INSTALL

We recommend that users build cudaLibMg samples using cmake. To build the complete samples, execute below:
```{engine='bash', count_lines}
mkdir build
cd build
cmake ..
make install -j4
```
