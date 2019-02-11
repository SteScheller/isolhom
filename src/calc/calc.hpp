#pragma once

#include <vector>
#include <array>
#include <iostream>

namespace calc
{
    //-------------------------------------------------------------------------
    // Function declarations
    //-------------------------------------------------------------------------
    /**
     * \brief calculates voxelwise local skew and curtosis for a given volume
     *
     * \param volumeData    pointer to the raw volume dta
     * \param volumeDim     three dimensional extents of the volume
     * \param windowDim     three dimensional extents of the local window
     *                      considered for the calculation of skew and curtosis
     */
    template<typename T>
    std::vector<std::array<double, 2>> calcLHOM(
        T* volumeData,
        std::array<size_t, 3> volumeDim,
        std::array<size_t, 3> windowDim);

    /**
     * \brief puts 2D data into a grid of bins
     *
     * \param data      pointer to vector of pairs containing the x and y
     *                  coordinates of the data points
     * \param binSize   pair containing the binSize in x and y direction
     * \param limits    pair of pairs containing the limits x and y dimension,
     *                  values outside of this range are discarded
     *
     * \return a two dimensional container whose elements contain the number
     * of data points that fall into the the according bin
     */
    template<typename T>
    void binning2D(
        std::vector<std::pair<T, T>> const & data,
        std::pair<T, T> binSize,
        std::pair<std::pair<T, T>, std::pair<T, T>> limits)
    {
        // initialize bins

        // TODO: print limits to check if indexing is correct
        std::cout << "X limits: " << limits[0][0] << ", " << limits[0][1] << std::endl;
        std::cout << "Y limits: " << limits[1][0] << ", " << limits[1][1] << std::endl;

        // iterate over values and count them in bins
        std::pair<T, T> item = { static_cast<T>(0), static_cast<T>(0) };
        size_t xBinNumber = 0, yBinNumber = 0;
        for (auto it = data.cbegin(); it != data.cend(); ++it)
        {
            item = *it;
            if (    (item[0] < limits[0][0]) ||
                    (item[0] > limits[0][1]) ||
                    (item[1] < limits[1][0]) ||
                    (item[1] > limits[1][1])    )
            {
                // item is outside of the given spatial limits
                continue;
            }

            // TODO
            xBinNumber = std::floor(item[0] / binSize[0]);
            xBinNumber = std::floor(item[1] / binSize[1]);

        }
    }

}
