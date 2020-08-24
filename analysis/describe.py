#! /usr/bin/env python3

import json
import gzip
import sys

def load(path):
    with gzip.open(path) as f:
        return json.load(f)

def describe(specimen):
    specimen = load(specimen)
    print(specimen['description'])



if __name__ == "__main__":
    if len(sys.argv) > 1:
        describe(sys.argv[1])
    else:
        print(f"Usage {sys.argv[0]} <specimen.json.gz>")
