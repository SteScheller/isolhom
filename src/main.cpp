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

#include <gnuplot-iostream.h>

#include "configraw/configraw.hpp"
#include "calc/calc.hpp"

//-----------------------------------------------------------------------------
// function prototypes
//-----------------------------------------------------------------------------
cr::VolumeConfig applyProgramOptions(int argc, char *argv[]);

//-----------------------------------------------------------------------------
// function implementations
//-----------------------------------------------------------------------------
/**
 * \brief Main entry point of the program
 */
int main(int argc, char *argv[])
{
    cr::VolumeConfig volumeConfig = applyProgramOptions(argc, argv);

    // check loaded volume
    std::cout << "Volume state: " << (volumeConfig.isValid() ? "valid" : "not valid")
        << std::endl;
    std::cout << "Number of timesteps: " << volumeConfig.getNumTimesteps() <<
        std::endl;
    if (!volumeConfig.isValid())
        return EXIT_FAILURE;

    // check if CUDA is available
    std::cout << "Running CUDA example code:" << std::endl;
    std::cout << calc::example<int>() << std::endl;

    // calculate skew and kurtosis
    // TODO: write loop and iterate over timesteps
    std::cout << "Calculating LHOMs" << std::endl;
    std::unique_ptr<cr::VolumeDataBase> volumeData;
    volumeData = cr::loadScalarVolumeTimestep(volumeConfig, 42, false);

    std::vector<std::array<float, 2>> lhoms;
    /*lhoms = calc::calcLHOM<unsigned_byte_t>(
        reinterpret_cast<unsigned_byte_t*>(volumeData->getRawData()),
        volumeConfig.getVolumeDim(),
        {5, 5, 5});*/
    std::vector<unsigned_byte_t> dummy(64*32*16, 0);
    lhoms = calc::calcLHOM<unsigned_byte_t>(
        dummy.data(),
        {64, 32, 16},
        {5, 5, 5});

    std::array<float, 3> sample;
    size_t idx = 0;
    sample = lhoms[idx];
    std::cout << "x = y = z = 0: " << sample[0] << " " << sample[1] << " " << sample[2] << " " << std::endl;
    idx = 15 * 32 * 64 + 31 * 64 + 63;
    sample = lhoms[idx];
    std::cout << "x = 63, y = 31, z = 15: " << sample[0] << " " << sample[1] << " " << sample[2] << " " << std::endl;

    // TODO: write stuff to csv or something similar

    return EXIT_SUCCESS;
}

/**
 * \brief Takes in input arguments, parses and loads the specified data
 *
 * \param   argc number of input arguments
 * \param   argv array of char pointers to the input arguments
 *
 * \return  a data object constructed from the input arguments
 */
cr::VolumeConfig applyProgramOptions(int argc, char *argv[])
{
    // Declare the supporded options
    po::options_description generic("Generic options");
    generic.add_options()
        ("help,h", "produce help message")
    ;

    po::options_description hidden("Hidden options");
    hidden.add_options()
        ("input-file",
         po::value<std::string>(),
         "json volume data description file")
    ;

    po::positional_options_description p;
    p.add("input-file", 1);

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
            std::cout << visible << std::endl;
            exit(EXIT_SUCCESS);
        }
        if (vm.count("input-file") != 1)
        {
            std::cout << "No input-file given!\n" << std::endl;
            std::cout << "Usage: isolhom [options] INPUT-FILE \n" << std::endl;
            std::cout << visible << std::endl;
            exit(EXIT_FAILURE);
        }

        po::notify(vm);

    }
    catch(std::exception &e)
    {
        std::cout << "Invalid program options!" << std::endl;
        exit(EXIT_FAILURE);
    }

    return cr::VolumeConfig((vm["input-file"].as<std::string>()));
}

