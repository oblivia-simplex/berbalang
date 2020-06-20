#! /usr/bin/env python3

import glob
import sys
import fire
import toml
import os
from datetime import datetime
import pytz



def figure_out_data_dir(iteration, data_root, population_name, date=None):
    if date is None:
        tz = pytz.timezone("America/Halifax")
        date = datetime.now(tz).strftime("%Y/%m/%d")
    # TODO Make this more flexible if need be
    population_name = f"{population_name}-{iteration}"
    dir = f"{data_root}/berbalang/Roper/Tournament/{date}/{population_name}"
    while os.path.exists(dir):
        population_name += "x"
        dir += "x"
    return (population_name, dir)



def sequential_runs(config_paths, num_trials, data_root):
    for config in config_paths:
        parsed = toml.load(config)
        name = parsed['population_name']
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
