#include <iostream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <array>

#include <cuda.h>

#include "calc.hpp"

//-----------------------------------------------------------------------------
// CUDA Kernels
//-----------------------------------------------------------------------------
// volume coordinate offset function with boundary handling (clamp to edge)
__device__
int3 offsetVolumeCoordinate(int3 coord, int3 offset, dim3 volumeDim)
{
    int3 dim = make_int3(
            (int) volumeDim.x, (int) volumeDim.y, (int) volumeDim.z);
    int3 coordOffset = make_int3(
        coord.x + offset.x, coord.y + offset.y, coord.z + offset.z);

    if(coordOffset.x < 0) coordOffset.x = 0;
    else if(coordOffset.x >= dim.x) coordOffset.x = volumeDim.x - 1;
    if(coordOffset.y < 0) coordOffset.y = 0;
    else if(coordOffset.y >= dim.y) coordOffset.y = volumeDim.y - 1;
    if(coordOffset.z < 0) coordOffset.z = 0;
    else if(coordOffset.z >= dim.z) coordOffset.z = volumeDim.z - 1;

    return coordOffset;
}

// kernel for calculating the local skew and kurtosis within the volume
template<typename T>
__global__
void lhom(
        double* results,
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
    if (    ((unsigned int) volCoord.x >= volumeDim.x) ||
            ((unsigned int) volCoord.y >= volumeDim.y) ||
            ((unsigned int) volCoord.z >= volumeDim.z) )
        return;

    // iterate over window block and calculate local higher order moments
    int3 wMid = make_int3(
            windowDim.x >> 1, windowDim.y >> 1, windowDim.z >> 1);
    int centerIdx =
            volCoord.z * volumeDim.y * volumeDim.x +
            volCoord.y * volumeDim.x +
            volCoord.x;
    double mean = 0.0;
    double kDiv = 1.0 / ((double) (windowDim.x * windowDim.y * windowDim.z));
    for (unsigned int wz = 0; wz < windowDim.z; ++wz)
    for (unsigned int wy = 0; wy < windowDim.y; ++wy)
    for (unsigned int wx = 0; wx < windowDim.x; ++wx)
    {
        int3 offset = make_int3(wx - wMid.x, wy - wMid.y, wz - wMid.z);
        int3 volCoordOffset =
            offsetVolumeCoordinate(volCoord, offset, volumeDim);
        int vIdx =
            volCoordOffset.z * volumeDim.y * volumeDim.x +
            volCoordOffset.y * volumeDim.x +
            volCoordOffset.x;
        mean += kDiv * ((double) volumeData[vIdx]);
    }

    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (unsigned int wz = 0; wz < windowDim.z; ++wz)
    for (unsigned int wy = 0; wy < windowDim.y; ++wy)
    for (unsigned int wx = 0; wx < windowDim.x; ++wx)
    {
        int3 offset = make_int3(wx - wMid.x, wy - wMid.y, wz - wMid.z);
        int3 volCoordOffset =
            offsetVolumeCoordinate(volCoord, offset, volumeDim);
        int vIdx =
            volCoordOffset.z * volumeDim.y * volumeDim.x +
            volCoordOffset.y * volumeDim.x +
            volCoordOffset.x;
        double val = (double) volumeData[vIdx];
        m2 += kDiv * pow((val - mean), 2.0);
        m3 += kDiv * pow((val - mean), 3.0);
        m4 += kDiv * pow((val - mean), 4.0);
    }

    // calculate skew and kurtosis and store them
    double skew = m3 / pow(m2, 1.5);
    double kurtosis = (m4 - 3.0 * pow(m2, 2.0)) / pow(m2, 2.0);
    results[centerIdx << 1] = skew;
    results[(centerIdx << 1) + 1] = kurtosis;
}


//-----------------------------------------------------------------------------
// Host wrapper functions
//-----------------------------------------------------------------------------
template<typename T>
std::vector<std::array<double, 2>> calc::calcLHOM(
    T* volumeData,
    std::array<size_t, 3> volumeDim,
    std::array<size_t, 3> windowDim)
{
    size_t n = volumeDim[0] * volumeDim[1] * volumeDim[2];
    std::vector<std::array<double, 2>> results(n, {0.0, -2.0});

    size_t resultDevMemSize = n * 2 * sizeof(double);
    double* resultDevMem = nullptr;
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

    numBlocks.x = static_cast<unsigned int>(0.5 +
            std::ceil(static_cast<double>(volumeDim[0]) /
                static_cast<double>(numThreads.x)));
    numBlocks.y = static_cast<unsigned int>(0.5 +
            std::ceil(static_cast<double>(volumeDim[1]) /
                static_cast<double>(numThreads.y)));
    numBlocks.z = static_cast<unsigned int>(0.5 +
            std::ceil(static_cast<double>(volumeDim[2]) /
                static_cast<double>(numThreads.z)));

    lhom<T><<<numBlocks, numThreads>>>(
            resultDevMem,
            volumeDevMem,
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
    cudaFree(volumeDevMem);
    cudaFree(resultDevMem);

    return results;
}

//-----------------------------------------------------------------------------
// Instantiations of templated functions
//-----------------------------------------------------------------------------
template std::vector<std::array<double, 2>> calc::calcLHOM<uint8_t>(
    uint8_t *volumeData,
    std::array<size_t, 3> volumeDim,
    std::array<size_t, 3> windowDim);

