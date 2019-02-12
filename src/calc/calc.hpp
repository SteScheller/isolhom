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
    template<typename dataT, typename returnT>
    std::vector<std::array<returnT, 2>> calcLHOM(
        dataT* volumeData,
        std::array<size_t, 3> volumeDim,
        std::array<size_t, 3> windowDim);

    /**
     * \brief puts 2D data into a grid of bins
     *
     * \param data      pointer to vector of pairs containing the x and y
     *                  coordinates of the data points
     * \param numBins   pair containing the number of bins in x and y direction
     * \param limits    pair of arrays containing the limits x and y dimension,
     *                  values outside of this range are discarded
     *
     * \return a two dimensional container whose elements contain the number
     * of data points that fall into the the according bin
     */
    template<typename T1, typename T2>
    std::vector<std::vector<size_t>> binning2D(
        std::vector<std::pair<T1, T2>> const & data,
        std::array<size_t, 2> numBins,
        std::pair<std::array<T1, 2>, std::array<T2, 2>> limits)
    {
        // initialize bins

        // TODO: print limits to check if indexing is correct
        std::cout << "X limits: " << limits.first[0] << ", " << limits.first[1] << std::endl;
        std::cout << "Y limits: " << limits.second[0] << ", " << limits.second[1] << std::endl;

        // iterate over values and count them in bins
        std::pair<T1, T2> item = { static_cast<T1>(0), static_cast<T2>(0) };
        for (auto it = data.cbegin(); it != data.cend(); ++it)
        {
            item = *it;
            if (    (item.first < limits.first[0]) ||
                    (item.first > limits.first[1]) ||
                    (item.second < limits.second[0]) ||
                    (item.second > limits.second[1])    )
            {
                // item is outside of the given spatial limits
                continue;
            }

            // TODO

        }
    }

}
