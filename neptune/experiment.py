#! /usr/bin/env python3

import fire
import glob
import os
import pytz
import sys
import toml
from datetime import datetime

import neptune

# TODO: for each run, save a gzipped copy of the executable used.
# this will make things very easy to replicate
# or better, create an archive of versions used, md5summed, and
# symlinked to the relevant log directories

def figure_out_data_dir(iteration, data_root, population_name, date=None):
    if date is None:
        tz = pytz.timezone("America/Halifax")
        date = datetime.now(tz).strftime("%Y/%m/%d")
    # TODO Make this more flexible if need be
    data_root = os.path.expanduser(data_root)
    data_root = os.path.abspath(data_root)
    dir = ""
    base_name = population_name
    iteration -= 1
    while dir == "" or os.path.exists(dir):
        print(f"{dir} already exists. Trying {dir}...")
        iteration += 1
        population_name = f"{base_name}-{iteration}"
        dir = f"{data_root}/berbalang/Roper/Tournament/{date}/{population_name}"
    return population_name, dir


def base(path):
    f, _ = os.path.splitext(path)
    return os.path.basename(f)


def sequential_runs(config_paths, num_trials, data_root):
    for config in config_paths:
        if not config.endswith(".toml"):
            continue
        parsed = toml.load(config)
        name = parsed['population_name'] if 'population_name' in parsed else \
            base(config)
        for i in range(0, num_trials):
            population_name, data_dir = figure_out_data_dir(i, data_root, name)
            print(f"Running trial for {population_name}")
            print(f"Expecting data in {data_dir}")
            err = os.system(f"./start.sh {config} {population_name}")
            if err:
                sys.exit(err)


def runs_for_config_dir(dir, trials, log_to):
    configs = glob.glob(f"{dir}/*.toml")
    sequential_runs(
        config_paths=configs,
        num_trials=trials,
        data_root=log_to,
    )


fire.Fire(runs_for_config_dir)
