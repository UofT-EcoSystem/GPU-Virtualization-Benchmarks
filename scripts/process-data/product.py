import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import argparse
import cxxfilt
import re
import difflib

def main():
    # cmd parser
    parser = argparse.ArgumentParser('Find kernel combination for concurrent execution.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--infile', nargs = '+', required=True, 
            help='Input directory of nsight compute dumps')

    # parse arguments
    args = parser.parse_args()

    # read csv
    df_list = [pd.read_csv(f) for f in args.infile]
    
    df = pd.concat(df_list)

    df.to_csv('total.csv', index=False, sep=',')

    df_zero = []
    df_one = []

    print('number of kernels: {}'.format(df.shape[0]))

    for i in range(df.shape[0]):
        for j in range(i+1, df.shape[0]):
            amp = df.iloc[j]
            amp['Benchmark_Set'] = ' & ' + amp['Benchmark_Set']
            amp['Kernel'] = ' & ' + amp['Kernel']
            summation = df.iloc[i] + amp

            num = summation[2:]

            if num[num > 100].count() == 0:
                df_zero.append(summation)
            elif num[num > 100].count() == 1:
                df_one.append(summation)

    df_zero = pd.DataFrame(df_zero) 
    df_one = pd.DataFrame(df_one) 

    df_zero.to_csv('zero.csv', index=False, sep=',')
    df_one.to_csv('one.csv', index=False, sep=',')

if __name__ == '__main__':
    main()


