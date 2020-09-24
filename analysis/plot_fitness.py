#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from glob import glob

def pop_dir():
    return os.path.basename(os.getcwd())

# should be run from the population pop_directory
def plot_feature_for_islands(feature, x=None, csv="mean", smooth=0, scale="linear"):
    plt.clf()
    num_islands = len(glob("island_[0-9]*"))
    data = [pd.read_csv(f"./island_{i}/{csv}_statistics.csv", index_col=False) for i in range(0, num_islands)]
    for i in range(0, num_islands):
        y = data[i][feature] if smooth == 0 else filters.gaussian_filter1d(data[i][feature], sigma=smooth)
        if x is None:
            plt.plot(y, label = f"island {i}")
        else:
            plt.plot(data[i][x], y, label = f"island {i}")
        plt.yscale(scale)
    if x is not None:
        plt.xlabel(f"{x}")
    plt.ylabel(f"{feature}")
    pop_name = pop_dir()
    plt.title(f"{pop_name} population {csv}: {feature} by island") 
    plt.legend()
    return data

