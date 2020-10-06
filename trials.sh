#! /usr/bin/env bash 

ulimit -c unlimited

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd)"

cd $DIR

#nix-shell -p python39Packages.pytz python39Packages.toml --run "./analysis/experiment.py $*"
./analysis/experiment.py $@
