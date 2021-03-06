#include <iostream>
#include <cstdio>
#include <cstdint>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <iterator>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/multi_array.hpp>

#include <gnuplot-iostream.h>

#include <CImg.h>
namespace img = cimg_library;

#include "configraw/configraw.hpp"
#include "calc/calc.hpp"
#include "progbar/progbar.hpp"
#include "inferno.hpp"

//-----------------------------------------------------------------------------
// type definitions
//-----------------------------------------------------------------------------
using lhom_t = double;
using volume_t = unsigned_byte_t;

//-----------------------------------------------------------------------------
// function prototypes
//-----------------------------------------------------------------------------
int applyProgramOptions(
        int argc,
        char *argv[],
        std::string& input,
        std::string& output,
        bool& csvOutput,
        bool& displayPlots,
        bool& individual,
        size_t& windowSize,
        std::vector<size_t>& imgSize);
void writeSkewKurtosisToCsv(
        std::string const &path,
        std::vector<std::array<lhom_t, 2>> lhoms,
        volume_t* volumeData);
void plotSkewKurtosis(
        std::string const &outBasename,
        std::array<size_t, 2> imgSize,
        boost::multi_array<size_t, 2> const &skewBins,
        boost::multi_array<size_t, 2> const &kurtosisBins,
        bool displayPlots,
        std::array<lhom_t, 2> skewLimits);
std::array<std::array<lhom_t, 2>, 2> binLhoms(
        std::vector<std::array<lhom_t, 2>> lhoms,
        volume_t* volumeData,
        std::array<volume_t, 2> volumeDataLimits,
        std::array<size_t, 2> numBins,
        boost::multi_array<size_t, 2> &skewBins,
        boost::multi_array<size_t, 2> &kurtosisBins);
//-----------------------------------------------------------------------------
// function implementations
//-----------------------------------------------------------------------------
/**
 * \brief Main entry point of the program
 */
int main(int argc, char *argv[])
{
    std::string input("");
    std::string output("");
    bool csvOutput = false;
    bool displayPlots = false;
    bool individual = false;
    std::vector<size_t> imgSize(2,  256);
    size_t windowSize = 3;

    if (EXIT_FAILURE == applyProgramOptions(
            argc,
            argv,
            input,
            output,
            csvOutput,
            displayPlots,
            individual,
            windowSize,
            imgSize))
    {
        std::cout << "Error: could not parse program options" << std::endl;
        return EXIT_FAILURE;
    }

    cr::VolumeConfig volumeConfig(input);

    // check loaded volume
    if (!volumeConfig.isValid())
    {
        std::cout << "Error: loaded volume not valid" << std::endl;
        return EXIT_FAILURE;
    }

    // check existence of output directory and create it if it does not exist
    fs::path outDir;
    if (output != "")
    {
        outDir = fs::path(output);
        if (fs::exists(outDir))
        {
            if (!fs::is_directory(outDir))
            {
                std::cout << "Error: " << outDir
                    << " exists but is not a directory!" << std::endl;
                return EXIT_FAILURE;
            }
        }
        else
        {
            try
            {
                fs::create_directories(outDir);
            }
            catch(fs::filesystem_error &e)
            {
                std::cout << "Error: "
                    << " failed to create output directory!" << std::endl;
                std::cout << e.what() << std::endl;
                return EXIT_FAILURE;
            }
        }
    }


    // allocate storage for calculation results
    boost::multi_array<size_t, 2> skewBins(
            boost::extents[imgSize[0]][imgSize[1]]);
    boost::multi_array<size_t, 2> kurtosisBins(
            boost::extents[imgSize[0]][imgSize[1]]);
    std::fill_n(skewBins.data(), skewBins.num_elements(), 0);
    std::fill_n(kurtosisBins.data(), kurtosisBins.num_elements(), 0);

    std::array<size_t, 3> volumeDim = volumeConfig.getVolumeDim();
    size_t numTimesteps = volumeConfig.getNumTimesteps();

    // calculate skew and kurtosis
    std::cout << "Calculating LHOMs" << std::endl;
    util::ProgressBar progbar(50, numTimesteps);
    for (size_t i = 0; i < numTimesteps; ++i)
    {
        progbar.print();
        std::cout.flush();

        // load data of current timestep
        std::unique_ptr<cr::VolumeDataBase> volumeData;
        volumeData = cr::loadScalarVolumeTimestep(volumeConfig, i, false);

        // calculate local higher order moments and put them into 2D histograms
        std::vector<std::array<lhom_t, 2>> lhoms;
        lhoms = calc::calcLHOM<volume_t, lhom_t>(
            reinterpret_cast<unsigned_byte_t*>(volumeData->getRawData()),
            volumeDim,
            {windowSize, windowSize, windowSize});
        boost::multi_array<size_t, 2> skewBinsTimestep(
                boost::extents[imgSize[0]][imgSize[1]]);
        boost::multi_array<size_t, 2> kurtosisBinsTimestep(
                boost::extents[imgSize[0]][imgSize[1]]);

        std::array<std::array<lhom_t, 2>, 2> lhomLimits = binLhoms(
                lhoms,
                reinterpret_cast<volume_t*>(volumeData->getRawData()),
                { 0, 255 },
                { imgSize[0], imgSize[1] },
                skewBinsTimestep,
                kurtosisBinsTimestep);

        // handle superimposing all timesteps
        for (size_t idxY = 0; idxY < imgSize[1]; ++idxY)
            for (size_t idxX = 0; idxX < imgSize[0]; ++idxX)
            {
                skewBins[idxX][idxY] += skewBinsTimestep[idxX][idxY];
                kurtosisBins[idxX][idxY] += kurtosisBinsTimestep[idxX][idxY];
            }

        // create output
        if (output == "")
        {
            if (displayPlots)
            {
                if (individual == true)
                    plotSkewKurtosis(
                        "",
                        {imgSize[0], imgSize[1]},
                        skewBinsTimestep,
                        kurtosisBinsTimestep,
                        displayPlots,
                        lhomLimits[0]);
                else
                    plotSkewKurtosis(
                        "",
                        {imgSize[0], imgSize[1]},
                        skewBins,
                        kurtosisBins,
                        displayPlots,
                        lhomLimits[0]);
            }
        }
        else
        {
            if (individual)
            {
                fs::path outFile(outDir);
                outFile /= fs::path(volumeConfig.getTimestepFile(i)).stem();
                plotSkewKurtosis(
                    outFile.c_str(),
                    {imgSize[0], imgSize[1]},
                    skewBinsTimestep,
                    kurtosisBinsTimestep,
                    displayPlots,
                    lhomLimits[0]);
            }

            // plot superimposed bins
            if (i == (numTimesteps - 1))
            {
                fs::path outFile(outDir);
                outFile /= fs::path(input).stem();
                plotSkewKurtosis(
                    outFile.c_str(),
                    {imgSize[0], imgSize[1]},
                    skewBins,
                    kurtosisBins,
                    displayPlots,
                    lhomLimits[0]);
            }

            if (csvOutput)
            {
                fs::path outFile(outDir / fs::path(
                    std::string(
                        fs::path(volumeConfig.getTimestepFile(i)
                            ).stem().c_str()) +
                    std::string(".csv")));
                writeSkewKurtosisToCsv(
                    outFile.c_str(),
                    lhoms,
                    reinterpret_cast<volume_t*>(volumeData->getRawData()));
            }
        }

        ++progbar;
    }
    ++progbar;
    progbar.print();

    return EXIT_SUCCESS;
}

/**
 * \brief Takes in input arguments, parses and loads the specified data
 *
 * \param argc          number of input arguments
 * \param argv          array of char pointers to the input arguments
 * \param input         path to the input file
 * \param output        directory where output shall be written to
 * \param csvOuput      set true if csv files shall be written
 * \param displayPlots  set true if the skew and kurtosis plots shall be
 *                      shown in an interactive window
 * \param individual    set to true if output shall be computed for all
 *                      individual timesteps
 * \param imgSize       gets filled with the requested width and height
 *                      (in pixel) for the skew and kurtosis plots
 *
 * \return EXIT_SUCCESS or EXIT_FAILURE depending if propram options could
 * be parsed successfully
 */
int applyProgramOptions(
        int argc,
        char *argv[],
        std::string& input,
        std::string& output,
        bool& csvOutput,
        bool& displayPlots,
        bool& individual,
        size_t& windowSize,
        std::vector<size_t>& imgSize)
{
    // Declare the supported options
    po::options_description generic("Generic options");
    generic.add_options()
        ("help,h", "produce help message")
        ("display,d", "show the the skew and kurtosis plots in an "
                      "interactive window")
        ("timesteps,t", "compute ouput for individual timesteps instead of "
                        "the whole timeseries")
        ("csv,c", "output csv files of the skew and kurtosis values")
        ("img-size,i",
         po::value<std::vector<size_t>>(&imgSize)->multitoken(),
                       "dimensions of the created plot in pixel "
                       "(default: 256x256)")
        ("window-size,w",
         po::value<size_t>(&windowSize)->default_value(3),
                       "size of the window for the calculation of local "
                       "higher order moments (default: 3)")
    ;

    po::options_description hidden("Hidden options");
    hidden.add_options()
        ("input-file",
         po::value<std::string>(),
         "json volume data description file")
        ("output-directory",
         po::value<std::string>(),
         "output directory for calculation results")
    ;

    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("output-directory", 2);

    po::options_description all("All options");
    all.add(generic).add(hidden);

    po::options_description visible("Allowed options");
    visible.add(generic);

    po::variables_map vm;

    try
    {
        po::store(po::command_line_parser(argc, argv).
                  options(all).positional(p).run(), vm);
        po::store(po::parse_command_line(argc, argv, all), vm);

        if (vm.count("help"))
        {
            std::cout <<
                "Usage: isolhom [options] INPUT-FILE [OUTPUT-DIR] \n" <<
                std::endl;
            std::cout << hidden << std::endl;
            std::cout << visible << std::endl;
            exit(EXIT_SUCCESS);
        }
        if (vm.count("input-file") != 1)
        {
            std::cout << "No input-file given!\n" << std::endl;
            std::cout <<
                "Usage: isolhom [options] INPUT-FILE [OUTPUT-DIR] \n" <<
                std::endl;
            std::cout << visible << std::endl;
            exit(EXIT_FAILURE);
        }
        else
            input = vm["input-file"].as<std::string>();
        if (vm.count("output-directory") != 1)
        {
            std::cout << "No output-directory given. " <<
                "Results will not be stored on disk!\n" << std::endl;
        }
        else
            output = vm["output-directory"].as<std::string>();
        if (vm.count("display")) displayPlots = true;
        else displayPlots = false;
        if (vm.count("csv")) csvOutput = true;
        else csvOutput = false;
        if (vm.count("timesteps")) individual = true;
        else individual = false;

        po::notify(vm);

    }
    catch(std::exception &e)
    {
        std::cout << "Invalid program options!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/**
 * \brief write results of LHOM calculation to csv file
 * \param path          filesystem path where csv shall be written to
 * \param lhoms         the calculated local higher order statistical moments
 * \param volumeData    pointer to the voxel data
 */
void writeSkewKurtosisToCsv(
        std::string const &path,
        std::vector<std::array<lhom_t, 2>> lhoms,
        volume_t* volumeData)
{
    std::ofstream out(path);

    out << "x,skew,kurtosis" << std::endl;
    for(size_t i = 0; i < lhoms.size(); ++i)
    {
        out << std::to_string(volumeData[i]) << ","
            << lhoms[i][0] << ","
            << lhoms[i][1] << std::endl;
    }
    out.close();
}

/**
 * \brief   plots the 2D histograms of local skew and kurtosis against the
 *          datavalue of the center voxel
 *
 * \param outBasename   path and filename stem for created images (empty
 *                      string if no output shall be written)
 * \param imgSize       width and height of the plots in pixel
 * \param skewBins      binned local skew data
 * \param kurtosisBins  binnes local kurtosis data
 * \param displayPlots  true if the created plots shall be shown in an
 *                      interactive window
 * \param skewLimits    highest and lowest skew value for drawing of zero level
 */
void plotSkewKurtosis(
        std::string const &outBasename,
        std::array<size_t, 2> imgSize,
        boost::multi_array<size_t, 2> const &skewBins,
        boost::multi_array<size_t, 2> const &kurtosisBins,
        bool displayPlots,
        std::array<lhom_t, 2> skewLimits)
{
    size_t skewBinMax = 0, kurtosisBinMax = 0;
    for(size_t y = 0; y < imgSize[1]; ++y)
    for(size_t x = 0; x < imgSize[0]; ++x)
    {
        if (skewBins[x][y] > skewBinMax) skewBinMax = skewBins[x][y];
        if (kurtosisBins[x][y] > kurtosisBinMax)
            kurtosisBinMax = skewBins[x][y];
    }

    // transform the 2D histograms into normalized, colormapped images
    img::CImg<size_t> skewImg(
            skewBins.data(), imgSize[1], imgSize[0], 1, 1, false);
    img::CImg<size_t> kurtosisImg(
            kurtosisBins.data(), imgSize[1], imgSize[0], 1, 1, false);
    static std::array<uint8_t, 256 * 3> infernoData;
    static bool initialized = false;
    if (!initialized)
        for (size_t i = 0; i < 256; ++i)
        {
            infernoData[i] = INFERNO_UINT8_RGB_256[i * 3];
            infernoData[i + 256] = INFERNO_UINT8_RGB_256[i * 3 + 1];
            infernoData[i + 512] = INFERNO_UINT8_RGB_256[i * 3 + 2];
        }
    initialized = true;
    static img::CImg<uint8_t> infernoMap(
            infernoData.data(), 256, 1, 1, 3, true);
    skewImg.rotate(-90.0);
    skewImg = skewImg.get_log().cut(0, skewBinMax).normalize(
            0, 255).map(infernoMap, 1);
    kurtosisImg.rotate(-90.0);
    kurtosisImg = kurtosisImg.get_log().cut(0, kurtosisBinMax).normalize(
            0, 255).map(infernoMap, 1);
    {
        // draw zero line for skew plot
        size_t yZero = -skewLimits[0] / (skewLimits[1] - skewLimits[0]) * 255;
        const unsigned char lineColor[] = { 255, 255, 255 };
        skewImg.draw_line(0, yZero, imgSize[0] - 1, yZero, lineColor);
    }

    // optionally show the plots and write the accordings images
    if (displayPlots)
        (skewImg, kurtosisImg).display();
    if (outBasename != "")
    {
        skewImg.save_png((outBasename + std::string("_skew.png")).c_str());
        kurtosisImg.save_png(
            (outBasename + std::string("_kurtosis.png")).c_str());
    }
}

/**
 * \brief plots the local skew and kurtosis against the datavalue of the
 *        center voxel
 * \param lhoms             the calculated local higher order statistical
 *                          moments
 * \param volumeData        pointer to the voxel data
 * \param volumeDataLimits  min and max values of the volume data
 * \param numBins           number of bins along the dimensions 'voxel value'
 *                          and skew/ kurtosis value
 * \param skewBins          reference for returning skew histogram data
 * \param kurtosisBins      reference for returning kurtosis histogram data
 *
 * \return min and max values of the local skew and kurtosis
 */
std::array<std::array<lhom_t, 2>, 2> binLhoms(
        std::vector<std::array<lhom_t, 2>> lhoms,
        volume_t* volumeData,
        std::array<volume_t, 2> volumeDataLimits,
        std::array<size_t, 2> numBins,
        boost::multi_array<size_t, 2> &skewBins,
        boost::multi_array<size_t, 2> &kurtosisBins)
{
    // prepare data for binning function
    std::vector<std::pair<volume_t, lhom_t>> skew(
            lhoms.size(), {0, 0.0});
    std::vector<std::pair<volume_t, lhom_t>> kurtosis(
            lhoms.size(), {0, 0.0});
    std::array<lhom_t, 2> skewLimits = {0.0, 0.0};
    std::array<lhom_t, 2> kurtosisLimits = {0.0, 0.0};
    for(size_t i = 0; i < lhoms.size(); ++i)
    {
        skew[i].first = volumeData[i]; skew[i].second = lhoms[i][0];
        kurtosis[i].first = volumeData[i]; kurtosis[i].second = lhoms[i][1];

        if(lhoms[i][0] < skewLimits[0]) skewLimits[0] = lhoms[i][0];
        else if(lhoms[i][0] > skewLimits[1]) skewLimits[1] = lhoms[i][0];

        if(lhoms[i][1] < kurtosisLimits[0]) kurtosisLimits[0] = lhoms[i][1];
        else if(lhoms[i][1] > kurtosisLimits[1])
            kurtosisLimits[1] = lhoms[i][1];
    }

    // make 2D histograms of the data
    skewBins = calc::binning2D(
        skew,
        numBins,
        {   {volumeDataLimits[0],volumeDataLimits[1]},
            {skewLimits[0], skewLimits[1]}  });
    kurtosisBins = calc::binning2D(
        kurtosis,
        numBins,
        {   {volumeDataLimits[0],volumeDataLimits[1]},
            {kurtosisLimits[0], kurtosisLimits[1]}  });

    return { skewLimits, kurtosisLimits };
}

