#!/usr/bin/env python3
"""Print max|a-b| / max|b| for two raw float64 solution dumps (no numpy needed)."""
import sys
from array import array


def load(path):
    a = array("d")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


a, b = load(sys.argv[1]), load(sys.argv[2])
if len(a) != len(b):
    print("nan")
    sys.exit(0)
den = max(abs(v) for v in b) or 1.0
print("%.3e" % (max(abs(x - y) for x, y in zip(a, b)) / den))
