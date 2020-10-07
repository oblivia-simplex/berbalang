#! /usr/bin/env python3

import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.ndimage.filters as filters
import sys
from glob import glob
import seaborn

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
            plt.plot(data[i][x], y, label=f"island {i}")
            if f"stdev_{feature}" in data[i].columns:
                plt.fill_between(data[i][x], 

                             y-2*data[i][f"stdev_{feature}"], 
                             y+2*data[i][f"stdev_{feature}"], alpha=0.5)
        plt.yscale(scale)
    if x is not None:
        plt.xlabel(f"{x}")
    plt.ylabel(f"{feature}")
    pop_name = pop_dir()
    plt.title(f"{pop_name} population {csv}: {feature} by island")
    plt.legend()
    return data


if __name__ == "__main__":
    pop = os.path.expanduser(sys.argv[1])
    feature = sys.argv[2]
    csv = sys.argv[3] if len(sys.argv) > 3 else "mean"
    cur_dir = os.getcwd()
    os.chdir(pop)
    p_name = pop_dir()

    plt.figure(figsize=(8, 5.5))
    plot_feature_for_islands(feature, x="epoch", smooth=1, csv=csv)

    # plt.show()
    os.chdir(cur_dir)
    filename = f"{p_name}__{feature}_{csv}.png"
    plt.savefig(filename, format="png", bbox="tight")
