#include <iostream>
#include <cstdio>
#include <string>
#include <chrono>
#include <thread>
#include <array>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <gnuplot-iostream.h>

#include "configraw/configraw.hpp"

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
    cr::VolumeConfig data = applyProgramOptions(argc, argv);

    std::cout << "Volume state: " << (data.isValid() ? "valid" : "not valid")
        << std::endl;
    std::cout << "Number of timesteps: " << data.getNumTimesteps() <<
        std::endl;

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

