#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plotSkewKurtosis(path):
    fig = plt.figure()
    data = pd.read_csv(path)
    ax1 = data.plot(kind='scatter',x='x', y='skew')
    ax2 = data.plot(kind='scatter',x='x', y='kurtosis')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'csv',
        nargs=1,
        help='path to the csv file which contains the calculated LHOMs')
    args = parser.parse_args()

    plotSkewKurtosis(args.csv[0])

