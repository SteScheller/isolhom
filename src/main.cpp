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
cr::VolumeConfig applyProgramOptions(
        int argc, char *argv[], std::string& output);

//-----------------------------------------------------------------------------
// function implementations
//-----------------------------------------------------------------------------
/**
 * \brief Main entry point of the program
 */
int main(int argc, char *argv[])
{
    std::string output("");
    cr::VolumeConfig volumeConfig = applyProgramOptions(argc, argv, output);

    // check loaded volume
    std::cout << "Volume state: " << (volumeConfig.isValid() ? "valid" : "not valid")
        << std::endl;
    std::cout << "Number of timesteps: " << volumeConfig.getNumTimesteps() <<
        std::endl;
    if (!volumeConfig.isValid())
        return EXIT_FAILURE;

    // calculate skew and kurtosis
    std::cout << "Calculating LHOMs" << std::endl;
    float progress = 0.f;
    for (size_t i = 0; i < volumeConfig.getNumTimesteps(); ++i)
    {
        // print progress bar (https://stackoverflow.com/a/14539953/2546289)
        int barWidth = 30;
        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        progress +=
            1.f / static_cast<float>(volumeConfig.getNumTimesteps());

        // load data of current timestep
        std::unique_ptr<cr::VolumeDataBase> volumeData;
        volumeData = cr::loadScalarVolumeTimestep(volumeConfig, i, false);

        // calculate stuff
        std::vector<std::array<float, 2>> lhoms;
        lhoms = calc::calcLHOM<unsigned_byte_t>(
            reinterpret_cast<unsigned_byte_t*>(volumeData->getRawData()),
            volumeConfig.getVolumeDim(),
            {5, 5, 5});

        // TODO: write stuff to csv or something similar
    }
    std::cout << "[==============================] 100 % Done!" << std::endl;
    std::cout << output << std::endl;

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
        int argc, char *argv[], std::string& output)
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
            std::cout << visible << std::endl;
            exit(EXIT_SUCCESS);
        }
        if (vm.count("input-file") != 1)
        {
            std::cout << "No input-file given!\n" << std::endl;
            std::cout <<
                "Usage: isolhom [options] INPUT-FILE OUTPUT-DIR \n" <<
                std::endl;
            std::cout << visible << std::endl;
            exit(EXIT_FAILURE);
        }
        if (vm.count("output-directory") != 1)
        {
            std::cout << "No output-directory given!\n" << std::endl;
            std::cout <<
                "Usage: isolhom [options] INPUT-FILE OUTPUT-DIR \n" <<
                std::endl;
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

    output = vm["output-directory"].as<std::string>();
    return cr::VolumeConfig((vm["input-file"].as<std::string>()));
}

