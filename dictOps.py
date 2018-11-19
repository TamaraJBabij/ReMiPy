# -*- coding: utf-8 -*-

# Applies a function to each value in a dict and returns a new dict with the results
def mapDict(d, fn):
    return {k: fn(v) for k,v in d.items()}

# Applies a function to each matching pair of values from 2 dicts
def mapDicts(a, b, fn):
    return {k: fn(a[k], b[k]) for k in a.keys()}

# Adds matching values from 2 dicts with the same keys
def addDicts(a,b):
    return mapDicts(a, b, lambda x, y: x + y)

# Multiply all values in a dict by the same constant
def multiplyDict(d, c):
    return mapDict(d, lambda x: x * c)