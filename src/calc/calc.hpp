#pragma once

#include <vector>
#include <array>
#include <iostream>

#include <boost/multi_array.hpp>

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
    boost::multi_array<size_t, 2> binning2D(
        std::vector<std::pair<T1, T2>> const & data,
        std::array<size_t, 2> numBins,
        std::pair<std::array<T1, 2>, std::array<T2, 2>> limits)
    {
        boost::multi_array<size_t, 2> bins(
                boost::extents[numBins[0]][numBins[1]]);

        // iterate over values and count them in bins
        std::pair<T1, T2> item = { static_cast<T1>(0), static_cast<T2>(0) };
        size_t idxX = 0, idxY = 0;
        double binSizeX =
            static_cast<double>(limits.first[1] - limits.first[0]) /
            static_cast<double>(numBins[0]);
        double binSizeY =
            static_cast<double>(limits.second[1] - limits.second[0]) /
            static_cast<double>(numBins[1]);
        for (auto it = data.cbegin(); it != data.cend(); ++it)
        {
            item = *it;
            if (    (item.first < limits.first[0]) ||
                    (item.first > limits.first[1]) ||
                    (item.second < limits.second[0]) ||
                    (item.second > limits.second[1])    )
            {
                // item is outside of the given spatial limits
                //std::cout << item.first << ", " << item.second << std::endl;
                continue;
            }

            // Calculate indices of the according bin and count the item
            idxX = static_cast<size_t>(
                static_cast<double>(item.first - limits.first[0]) /
                binSizeX);
            idxY = static_cast<size_t>(
                static_cast<double>(item.second - limits.second[0]) /
                binSizeY);
            // upper limit is included in the last bin
            if (idxX >= numBins[0]) idxX = numBins[0] - 1;
            if (idxY >= numBins[1]) idxY = numBins[1] - 1;
            ++(bins[idxX][idxY]);
        }

        return bins;
    }

}
