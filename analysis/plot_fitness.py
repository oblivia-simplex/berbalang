#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from glob import glob

# should be run from the population directory
def plot_feature_for_islands(feature, x=None, csv="mean", smooth=0, scale="linear"):
    num_islands = len(glob("island_[0-9]*"))
    data = [pd.read_csv(f"./island_{i}/{csv}_statistics.csv") for i in range(0, num_islands)]
    for i in range(0, num_islands):
        y = data[i][feature] if smooth == 0 else filters.gaussian_filter1d(data[i][feature], sigma=smooth)
        if x is None:
            plt.plot(y)
        else:
            plt.plot(data[i][x], y)
        plt.yscale(scale)
    return data

