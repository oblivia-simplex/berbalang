#! /usr/bin/env bash

#####
# Checks to make sure a config is okay, by running berbalang on it for just 2 epochs
#####
set -e # break on first error

config_dir="$1"
if [ -z "$config_dir" ]; then
  echo "Usage $0 <config dir>"
  exit 1
fi

function check() {
  config="$1"
  r=`basename $config | sed "s/\.toml//"`
  tmp=`mktemp /tmp/${r}_XXX.toml`
  cat $config | sed "s/pop_size.*$/pop_size = 10/" | sed "s/num_epochs.*$/num_epochs = 1/" > $tmp

  ./run_with_trace.sh $tmp $tmp
}

for f in `find $config_dir -type f -name "*.toml" | shuf` ; do
  echo "[+] Checking $f"
  if ! check $f  ; then
    echo "[x] FAILED ON $f"
    exit 1
  fi
done
