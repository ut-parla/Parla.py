#ifndef SINGLETON_HPP
#define SINGLETON_HPP

#include <memory>
#include "cublas_v2.h"
#include <algorithm>
#include <numeric>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE
#include <assert.h>
#include <cstring>

template <typename T>
class Singleton
{
    friend class knnHandle_t; // access private constructor/destructor
    friend class mgpuHandle_t;

public:
    static T &instance()
    {
        static const std::unique_ptr<T> instance{new T()};
        return *instance;
    }

private:
    Singleton(){};
    ~Singleton(){};
    Singleton(const Singleton &) = delete;
    void operator=(const Singleton &) = delete;
};

class knnHandle_t final : public Singleton<knnHandle_t>
{
    friend class Singleton<knnHandle_t>; // access private constructor/destructor
private:
    knnHandle_t()
    {
        //std::cout<<"Create Handle_t instance"<<std::endl;
        // cublas handle
        cublasCreate(&blas);
    }

public:
    ~knnHandle_t()
    {
        //std::cout<<"Destroy knnHandle_t instance"<<std::endl;
        cublasDestroy(blas);
    }

public:
    cublasHandle_t blas;
};

#endif