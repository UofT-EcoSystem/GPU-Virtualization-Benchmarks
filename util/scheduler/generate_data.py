import random
import numpy as np
import scipy
import logging
import pandas as pd

# convert csv file to dataframe
slowdowns = pd.read_csv('slowdown.csv', sep=',', header=None)
print(slowdowns)

lengths = pd.read_csv('kernel_idx.csv', sep=',', header=None)
print(lengths)


# if __name__ == "__main__":
