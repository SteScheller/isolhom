#pragma once

#include <vector>
#include <array>

namespace calc
{
    //-------------------------------------------------------------------------
    // Function declarations
    //-------------------------------------------------------------------------
    template<typename T>
    T example();

    /**
     * \brief calculates voxelwise local skew and curtosis for a given volume
     *
     * \param volumeData    pointer to the raw volume dta
     * \param volumeDim     three dimensional extents of the volume
     * \param windowDim     three dimensional extents of the local window
     *                      considered for the calculation of skew and curtosis
     */
    template<typename T>
    std::vector<std::array<float, 3>> calcLHOM(
        T* volumeData,
        std::array<size_t, 3> volumeDim,
        std::array<size_t, 3> windowDim);
}
