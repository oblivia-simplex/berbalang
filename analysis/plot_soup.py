#! /usr/bin/env python3

import matplotlib.pyplot as plt
import os
import re
import pandas
import sys
import glob
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
parser.add_argument("--title", default="", help="title")

def get_soup_number(soup_path):
    base = os.path.basename(soup_path)
    nums = re.findall("[0-9]+", base)
    return int(nums[0])

def plot_soup(tsstart=0, tsend=-1, tsstep=1, scale="linear", title=""):
    plt.clf()
    # load the data
    soupfiles = glob.glob("soup/soup_*.json")
    assert(len(soupfiles)>0)
    soupindexes = sorted([get_soup_number(s) for s in soupfiles])
    soupdfs = []
    x_label = "timestep"
    for ix in soupindexes[tsstart:tsend:tsstep]:
        try:
            with open(f"soup/soup_{ix}.json") as f:
                soup = dict(json.load(f))
        except:
            with open(f"soup/soup_at_epoch_{ix}.json") as f:
                soup = dict(json.load(f))
                x_label = "epoch"
        soupdf = pandas.DataFrame({k:soup[k] for k in soup if soup[k] >= 100}, index=[ix])
        soupdfs.append(soupdf)
    soup = pandas.concat(soupdfs)
    soup.fillna(0)
    # build the plot
    plt.plot(soupindexes[tsstart:tsend:tsstep], soup, 'k-', alpha=0.1)
    plt.setp(plt.gca().get_xticklabels(), ha="right", rotation=30)
    plt.xlabel(x_label)
    plt.ylabel("occurrences")
    if title:
        plt.title(title)
    else:
        plt.title("Alleles in soup over time")
    ylim = os.getenv("PLOT_YLIM")
    if ylim:
        plt.ylim((0,float(ylim)))
    else:
        plt.yscale(scale)
    
if __name__ == "__main__":
    args = parser.parse_args()

    cur_dir = os.getcwd()
    os.chdir(args.path)

    plt.figure(figsize=(8, 5.5))
    with plt.style.context("seaborn"):
        plot_soup(args.start, args.end, args.step, title=args.title)

    # plt.show()
    os.chdir(cur_dir)
    filename = f"{args.slug}_soup.{args.format}"
    plt.tight_layout()
    plt.savefig(filename, format=args.format, bbox="tight")
