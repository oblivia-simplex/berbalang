#! /usr/bin/env python3

import gzip
import json
import functools as ft
import capstone


def load_specimen(path):
    try:
        with gzip.open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read specimen from {path}: {e}")
        return None


def get_code(specimen):
    return [bytes(code) for code in specimen["profile"]["code_executed"]]
