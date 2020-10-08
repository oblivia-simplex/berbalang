#! /usr/bin/env python3

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import gzip
import json
import glob
import os
import sys

def load_population(pop_path):
    try:
        with gzip.open(pop_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"failed: {e}")
        return []

def mkrow(c):
    return { 
        "generation": c["chromosome"]["generation"],
        "num_offspring": c["num_offspring"],
        "name": c["chromosome"]["name"],
        "code_coverage": c["fitness"]["scores"]["code_coverage"],
        "length": len(c["chromosome"]["chromosome"]),
        "ret_count": c["fitness"]["scores"]["ret_count"]
    }

def scrape_populations(pop_paths):
    rows = []
    for p in pop_paths:
        pop = load_population(p)
        print(f"processing {p}")
        for c in pop:
            rows.append(mkrow(c))
    return pd.DataFrame(list(rows))
    

def plot(data):
    with plt.style.context("seaborn"):
        ax = sns.relplot(x="generation", 
                y="ret_count",
                hue="code_coverage",
                size="num_offspring",
                data=data)
        return ax


def plot_population(pop_dir):
    populations = glob.glob(f"{pop_dir}/island_*/population/*.json.gz")
    data = scrape_populations(populations)
    plt.figure(figsize=(8, 5.5))
    plot(data)
    pop_basename = os.path.basename(pop_dir)
    filename = f"{pop_basename}_scatterplot.png"
    plt.savefig(filename, format="png", bbox="tight")
    plt.close()
    return

if __name__ == "__main__":
    plot_population(sys.argv[1])
