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
    data = pd.DataFrame(list(rows))
    return data

    

def plot(data):
    with plt.style.context("seaborn"):
        ax = sns.relplot(x="generation", 
                hue="ret_count",
                hue_norm=(0, 15),
                y="code_coverage",
                size="num_offspring",
                sizes=(15,200),
                alpha=0.7,
                data=data)
        ax.set(ylim=(0,0.0005))
        ax.set(xlim=(0,250))
        return ax


def plot_population(pop_dir):
    pop_basename = os.path.basename(pop_dir)
    print(f"Plotting {pop_basename} population...")
    populations = glob.glob(f"{pop_dir}/island_*/population/*.json.gz")
    scatter_csv = f"{pop_basename}_scatter_data.csv"
    data = None
    if os.path.exists(scatter_csv):
        print(f"reading data from {scatter_csv}...")
        data = pd.read_csv(scatter_csv, index_col=0)
    else:
        print(f"{scatter_csv} not found. scraping populations (slow)...")
        data = scrape_populations(populations)
        print(f"saving to {scatter_csv}")
        data.to_csv(scatter_csv)
    plt.figure(figsize=(8, 5.5))
    plot(data)
    plt.title(f"{pop_basename}")
    filename = f"{pop_basename}_scatterplot.png"
    plt.tight_layout(h_pad=2.0)
    plt.savefig(filename, format="png", bbox="tight")
    plt.close()
    return

if __name__ == "__main__":
    plot_population(sys.argv[1])
