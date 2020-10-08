#! /usr/bin/env python3

import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.ndimage.filters as filters
import sys
from glob import glob
import seaborn
import textwrap
import matplotlib



def pop_dir():
    return os.path.basename(os.getcwd())


def theme():
    t = os.getenv("PLOT_THEME")
    if t:
        return t
    else:
        return "seaborn"


# should be run from the population pop_directory
def plot_feature_for_islands(feature, x=None, csv="mean", smooth=0, scale="linear"):
    plt.clf()
    num_islands = len(glob("island_[0-9]*"))
    data = [pd.read_csv(f"./island_{i}/{csv}_statistics.csv", index_col=False) for i in range(0, num_islands)]
    with plt.style.context(theme()):
        for i in range(0, num_islands):
            y = data[i][feature] if smooth == 0 else filters.gaussian_filter1d(data[i][feature], sigma=smooth)
            if x is None:
                plt.plot(y, label = f"island {i}")
            else:
                plt.plot(data[i][x], y, label=f"island {i}")
                if f"stdev_{feature}" in data[i].columns:
                    plt.fill_between(data[i][x], 
                                 y-2*data[i][f"stdev_{feature}"], 
                                 y+2*data[i][f"stdev_{feature}"], alpha=0.2)
            plt.yscale(scale)
        if x is not None:
            plt.xlabel(f"{x}")
        ylim = os.getenv("PLOT_YLIM")
        if ylim:
            plt.ylim(0, float(ylim))
        plt.ylabel(f"{feature}")
        pop_name = pop_dir().capitalize()
        plt.title(f"{csv} {feature} in the\n{pop_name} population".capitalize())
        plt.legend()
    return data


def plot_all_features(path_to_population_dir, smooth=1):
    cur_dir = os.getcwd()
    os.chdir(path_to_population_dir)
    pop_name = pop_dir()
    # get the headers to see what features we want
    for csv in ["mean", "champion"]:
        with open(f"island_0/{csv}_statistics.csv", "r") as f:
            features = [x for x in f.readline().strip().split(",")[1:] if not x.startswith("stdev_")]
        for feature in features:
            print(f"Plotting {csv} {feature} for {pop_name}...")
            plt.figure(figsize=(8, 5.5))
            plot_feature_for_islands(feature, x="epoch", csv=csv, smooth=smooth, scale="linear")
            filename = f"{cur_dir}/{pop_name}__{feature}_{csv}.png"
            plt.savefig(filename, format="png", bbox="tight")
            plt.close()
    return





if __name__ == "__main__":
    pop = os.path.expanduser(sys.argv[1])
    feature = sys.argv[2] if len(sys.argv) > 2 else "all"
    csv = sys.argv[3] if len(sys.argv) > 3 else "mean"
    
    if feature == "all":
        plot_all_features(pop)
    else:
        cur_dir = os.getcwd()
        os.chdir(pop)
        p_name = pop_dir()

        plt.figure(figsize=(8, 5.5))
        plot_feature_for_islands(feature, x="epoch", smooth=1, csv=csv)

        # plt.show()
        os.chdir(cur_dir)
        filename = f"{p_name}__{feature}_{csv}.svg"
        plt.savefig(filename, format="svg", bbox="tight")
