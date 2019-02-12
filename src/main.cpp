#include <iostream>
#include <cstdio>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <array>
#include <memory>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <gnuplot-iostream.h>

#include "configraw/configraw.hpp"
#include "calc/calc.hpp"
#include "progbar/progbar.hpp"

//-----------------------------------------------------------------------------
// type definitions
//-----------------------------------------------------------------------------
using lhom_t = double;
using volume_t = unsigned_byte_t;

//-----------------------------------------------------------------------------
// function prototypes
//-----------------------------------------------------------------------------
cr::VolumeConfig applyProgramOptions(
        int argc, char *argv[], std::string& output, bool& imageOutput);
void writeToCsv(
        std::string const &path,
        std::vector<std::array<lhom_t, 2>> lhoms,
        volume_t* volumeData);
void writeToPng(
        std::string const &outBasename,
        std::vector<std::array<lhom_t, 2>> lhoms,
        volume_t* volumeData);

//-----------------------------------------------------------------------------
// function implementations
//-----------------------------------------------------------------------------
/**
 * \brief Main entry point of the program
 */
int main(int argc, char *argv[])
{
    std::string output("");
    bool imageOutput = false;

    cr::VolumeConfig volumeConfig = applyProgramOptions(
            argc, argv, output, imageOutput);

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

    // calculate skew and kurtosis
    std::cout << "Calculating LHOMs" << std::endl;
    util::ProgressBar progbar(50, volumeConfig.getNumTimesteps());
    for (size_t i = 0; i < volumeConfig.getNumTimesteps(); ++i)
    {
        progbar.print();
        std::cout.flush();

        // load data of current timestep
        std::unique_ptr<cr::VolumeDataBase> volumeData;
        volumeData = cr::loadScalarVolumeTimestep(volumeConfig, i, false);

        // calculate local higher order moments
        std::vector<std::array<lhom_t, 2>> lhoms;
        lhoms = calc::calcLHOM<volume_t, lhom_t>(
            reinterpret_cast<unsigned_byte_t*>(volumeData->getRawData()),
            volumeConfig.getVolumeDim(),
            {5, 5, 5});

        // write image or csv output
        if (output != "")
        {
            if (!imageOutput)
            {
                fs::path outFile(outDir / fs::path(
                    std::string(
                        fs::path(volumeConfig.getTimestepFile(i)
                            ).stem().c_str()) +
                    std::string(".csv")));
                writeToCsv(
                    outFile.c_str(),
                    lhoms,
                    reinterpret_cast<unsigned_byte_t*>(
                        volumeData->getRawData()));
            }
            else
            {
                // TODO: Output the calculated lhoms as images


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
 * \param   argc number of input arguments
 * \param   argv array of char pointers to the input arguments
 * \param   path to directory where output shall be written to
 *
 * \return  a data object constructed from the input arguments
 */
cr::VolumeConfig applyProgramOptions(
        int argc, char *argv[], std::string& output, bool& imageOutput)
{
    // Declare the supporded options
    po::options_description generic("Generic options");
    generic.add_options()
        ("help,h", "produce help message")
        ("image,i", "output images of the skew and kurtosis vs. data values "
                    "plots instead of csv files")
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
            exit(EXIT_FAILURE);
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
        if (vm.count("output-directory") != 1)
        {
            std::cout << "No output-directory given. " <<
                "Results will not be stored on disk!\n" << std::endl;
        }
        else
            output = vm["output-directory"].as<std::string>();
        if (vm.count("image"))
            imageOutput = true;
        else
            imageOutput = false;

        po::notify(vm);

    }
    catch(std::exception &e)
    {
        std::cout << "Invalid program options!" << std::endl;
        exit(EXIT_FAILURE);
    }

    return cr::VolumeConfig((vm["input-file"].as<std::string>()));
}

/**
 * \brief write results of LHOM calculation to csv file
 *
 */
void writeToCsv(
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

void writeToPng(
        std::string const &outBasename,
        std::vector<std::array<lhom_t, 2>> lhoms,
        volume_t* volumeData)
{
    std::vector<std::pair<volume_t, lhom_t>> skew(
            lhoms.size(), {0, 0.0});
    std::vector<std::pair<volume_t, lhom_t>> kurtosis(
            lhoms.size(), {0, 0.0});
    std::pair<lhom_t, lhom_t> skewLimits = {0.0, 0.0};
    std::pair<lhom_t, lhom_t> kurtosisLimits = {0.0, 0.0};
    for(size_t i = 0; i < lhoms.size(); ++i)
    {
        skew[i].first = volumeData[i];
        skew[i].second = lhoms[i][0];
        kurtosis[i].first = volumeData[i];
        kurtosis[i].second = lhoms[i][1];
    }

    std::vector<std::vector<size_t>> skewBins = calc::binning2D(
        skew,
        {256, 256},
        {{0, 255}, {}});

    fs::path outSkew(outBasename + std::string("_skew.png"));
    fs::path outKurtosis(outBasename + std::string("_kurtosis.png"));
}
