"""This module provides compilation flags used by ycm to enable
semantic code completion."""

import os
import sys

DIR_OF_THIS_SCRIPT = os.path.abspath(os.path.dirname( __file__ ))

INCLUDES = [
        'src',
        'include',
        'lib/gnuplot-iostream',
        'lib/nlohmann',
        'src/configraw',
        'src/calc']
FLAGS = [
        '-std=c++14',
        '-Wall', '-Wextra']
LIBS = [
        '-lboost_program_options', '-lboost_system', '-lboost_iostreams']

# for debugging
log = None

def FlagsForFile(filename, **kwargs):
    includes = []
    for item in INCLUDES:
        includes.append('-I' + os.path.join(DIR_OF_THIS_SCRIPT, item))


    if log is not None:
        orig_stdout = sys.stdout
        with open(log, 'w') as f:
            sys.stdout = f
            print(kwargs)
            print(kwargs['client_data'])
            for item in kwargs['client_data']:
                print(item, kwargs['client_data'][item])
            print(filename)
            print(includes)
        sys.stdout = orig_stdout

    ctype = []

    return { 'flags': FLAGS + ctype + includes + LIBS }


