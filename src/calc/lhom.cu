#include <iostream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <array>

#include <cuda.h>

#include "calc.hpp"

//-----------------------------------------------------------------------------
// Example code for reference
//
// adapted from https://devblogs.nvidia.com/even-easier-introduction-cuda/
//-----------------------------------------------------------------------------
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

template<typename T>
T calc::example()
{
    T ret = 0;
    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = std::fmax(maxError, std::fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return ret;
}

template int calc::example<int>();
//-----------------------------------------------------------------------------
// volume coordinate offset function including boundary handling
//
/*__device__
int3 offsetVolumeCoordinate(int3 coord, int3 offset, uint3 volumeDim)
{
    size_t xn = x;
    long long tmp = 0;

    tmp = static_cast<long long>(x) + offset;
    if ( (tmp < 0LL) || (tmp >= static_cast<long long>(size)))
    {

        // apply border policy
        switch (bp)
        {
            case BorderPolicy::mirror:
                tmp = tmp < 0LL ? (-1LL * tmp) - 1LL : tmp;
                xn = tmp > static_cast<long long>(size - 1)?
                    static_cast<size_t>(size) +
                        ((size - 1) - static_cast<size_t>(tmp)) :
                    static_cast<size_t>(tmp);
                break;

            case BorderPolicy::repeat:
                xn = (tmp >= 0) ? tmp % size: size - (-tmp % size);
                break;

            case BorderPolicy::clamp_to_edge:
            default:
                xn = std::min(
                        static_cast<size_t>(std::max(tmp, 0LL)), size - 1);
                break;
        }
    }
    else
        xn = static_cast<size_t>(tmp);

    return xn;
}*/

// kernel for calculating the local skew and kurtosis within the volume
template<typename T>
__global__
void lhom(
        float* results, 
        T* volumeData, 
        dim3 volumeDim, 
        dim3 windowDim)
{
    // calculate coordinate within volume from thread and block index
    int3 volCoord = make_int3(
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z);

    // if we are outside the volume we dont have to do anything
    if (    (volCoord.x >= volumeDim.x) ||
            (volCoord.y >= volumeDim.y) ||
            (volCoord.z >= volumeDim.z) )
        return;

    // iterate over window block
    int3 wMid = make_int3(
            windowDim.x >> 1, windowDim.y >> 1, windowDim.z >> 1);
    int centerIdx = 
            volCoord.z * volumeDim.y * volumeDim.x +
            volCoord.y * volumeDim.x + 
            volCoord.x;
    T midValue = volumeData[centerIdx];
    /*for (unsigned int wz = 0; wz < windowDim.z; ++wz)
    for (unsigned int wy = 0; wy < windowDim.y; ++wy)
    for (unsigned int wx = 0; wx < windowDim.x; ++wx)
    {
        int3 offset = make_int3(wx - wMid.x, wy - wMid.y, wz - wMid.z);
    }*/
    results[centerIdx * 3] = volCoord.x;
    results[centerIdx * 3 + 1] = volCoord.y;
    results[centerIdx * 3 + 2] = volCoord.z;
}


template<typename T>
std::vector<std::array<float, 3>> calc::calcLHOM(
    T* volumeData,
    std::array<size_t, 3> volumeDim,
    std::array<size_t, 3> windowDim)
{
    size_t n = volumeDim[0] * volumeDim[1] * volumeDim[2];
    std::vector<std::array<float, 3>> results(n, {0.f, -2.f, 0.f});

    size_t resultDevMemSize = n * 3 * sizeof(float);
    float* resultDevMem = nullptr;
    size_t volumeDevMemSize = n * sizeof(T);
    T* volumeDevMem = nullptr;

    cudaMalloc(&volumeDevMem, volumeDevMemSize);
    cudaMalloc(&resultDevMem, resultDevMemSize);
    cudaMemcpy(
            volumeDevMem,
            volumeData,
            volumeDevMemSize,
            cudaMemcpyHostToDevice);

    // calculate number of threads and block and invoke the kernel
    dim3 numBlocks;
    dim3 numThreads(8, 8, 8);

    numBlocks.x = static_cast<unsigned int>(0.5f +
            std::ceil(static_cast<float>(volumeDim[0]) / 
                static_cast<float>(numThreads.x)));
    numBlocks.y = static_cast<unsigned int>(0.5f + 
            std::ceil(static_cast<float>(volumeDim[1]) / 
                static_cast<float>(numThreads.y)));
    numBlocks.z = static_cast<unsigned int>(0.5f + 
            std::ceil(static_cast<float>(volumeDim[2]) / 
                static_cast<float>(numThreads.z)));

    lhom<T><<<numBlocks, numThreads>>>(
            resultDevMem,
            volumeData,
            {   static_cast<unsigned int>(volumeDim[0]), 
                static_cast<unsigned int>(volumeDim[1]), 
                static_cast<unsigned int>(volumeDim[2]) },
            {   static_cast<unsigned int>(windowDim[0]), 
                static_cast<unsigned int>(windowDim[1]), 
                static_cast<unsigned int>(windowDim[2]) });

    // Wait for GPU to finish and get the results afterwards
    cudaDeviceSynchronize();
    cudaMemcpy(
            results.data(),
            resultDevMem,
            resultDevMemSize,
            cudaMemcpyDeviceToHost);
    cudaFree(resultDevMem);

    return results;
}

template std::vector<std::array<float, 3>> calc::calcLHOM<uint8_t>(
    uint8_t *volumeData,
    std::array<size_t, 3> volumeDim,
    std::array<size_t, 3> windowDim);

