import uproot
from pathlib import Path
import numpy as np
import pandas as pd

# Open a single tree from all files in a directory
# path: The path of the directory to load from
# glob (optional): The pattern to match files against, defaults to anything
#   with the .root extension
# tree (optional): The name of the tree to load from each file, defaults to T
def openFromDirectory(path, glob="*.root", tree="T"):
    return (openFromFile(f) for f in Path(path).glob(glob))


# Open a tree from a file
# path: The path of the file to load from
# tree (optional): The name of the tree to load, defaults to T
def openFromFile(path, tree="T"):
    return uproot.open(f"file://{path}")[tree]

def as2DArray(tree):
    return np.column_stack(tree.arrays().values())

def asDataFrame(trees):
    return pd.concat(tree.pandas.df() for tree in trees)