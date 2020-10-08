#! /usr/bin/env python3

import matplotlib.pyplot as plt
import os
import pandas
import sys
import glob
import seaborn
import simplejson as json

import argparse

parser = argparse.ArgumentParser(description="""
Plot the allele history of a soup directory.
Run this program from within the island subdirectory -- or pass the island subdirectory as a path.
""")
parser.add_argument("--path", default=".", help="path to the relevant island subdirectory")
parser.add_argument("--start", type=int, default=0, help="starting timestep")
parser.add_argument("--end", type=int, default=-1, help="ending timestep")
parser.add_argument("--step", type=int, default=1, help="timestep stepsize")
parser.add_argument("--slug", default="", help="slug for distinguishing different plots")
parser.add_argument("--format", default="png", help="format for saving the resulting figure")

def plot_soup(tsstart=0, tsend=-1, tsstep=1, scale="linear"):
    plt.clf()
    # load the data
    soupfiles = glob.glob("soup/soup_*.json")
    assert(len(soupfiles)>0)
    soupindexes = sorted([int(t[5:].split(".")[0]) for s in soupfiles for t in [os.path.basename(s)]])
    soupdfs = []
    for ix in soupindexes[tsstart:tsend:tsstep]:
        with open(f"soup/soup_{ix}.json") as f:
            soup = dict(json.load(f))
        soupdf = pandas.DataFrame({k:soup[k] for k in soup if soup[k] >= 100}, index=[ix])
        soupdfs.append(soupdf)
    soup = pandas.concat(soupdfs)
    soup.fillna(0)
    # build the plot
    plt.plot(soupindexes[tsstart:tsend:tsstep], soup, 'k-', alpha=0.1)
    plt.setp(plt.gca().get_xticklabels(), ha="right", rotation=30)
    plt.xlabel("timestep")
    plt.ylabel("# instances")
    plt.title("Alleles in soup over time")
    plt.yscale(scale)
    
if __name__ == "__main__":
    args = parser.parse_args()

    cur_dir = os.getcwd()
    os.chdir(args.path)

    plt.figure(figsize=(8, 5.5))
    plot_soup(args.start, args.end, args.step)

    # plt.show()
    os.chdir(cur_dir)
    filename = f"{args.slug}soup.{args.format}"
    plt.tight_layout()
    plt.savefig(filename, format=args.format, bbox="tight")
