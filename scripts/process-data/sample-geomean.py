#!/usr/bin/env python

from primitives.frontend import print_sys_perf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
import os.path
from textwrap import wrap
from scipy.stats import gmean
from matplotlib2tikz import save as tikz_save


plt.rcParams["figure.figsize"] = (15,15)

def plot_mps(bars, xticks, title, ylabel, ymax, ideal):
    # set width of bar
    barWidth = 0.25

    r = np.arange(len(bars))

    plt.bar(r, bars, color='#7f6d5f', edgecolor='white', 
            width=barWidth, label='MPS', capsize=3)

    plt.xticks(range(len(bars)), xticks, fontsize=18)

    plt.legend(fontsize=20)
       
    # Add xticks on the middle of the group bars
    plt.xlabel('Test Cases', fontsize=20)
         
    plt.title(title, fontsize=24)
    plt.ylabel(ylabel, fontsize=20)
    plt.yticks(fontsize=18)
    plt.axhline(ideal, zorder=3, color='black', linestyle='--')
    plt.ylim([0,ymax])
    plt.savefig(title + '.png', transparent=False)
    plt.close()

    #tikz_save(title + '.tex')
 


def main():
    if (len(sys.argv) < 2):
         print("Usage: <path/to/script> <path/to/config>...")
         sys.exit()

    ws_bars = []
    fairness_bars = []

    xticks = []
    for i in range(1, len(sys.argv)):
        config = sys.argv[i]

        group_name = os.path.basename(config).split('.')[0]
        xticks.append(group_name)

        res_time, res_mps = print_sys_perf(config, 
                  os.path.join(os.path.dirname(os.path.abspath(config)), 'experiments'))
        # mps only
        res_mps = np.array(res_mps)

        # get weighted speedup plot
        ws = gmean(res_mps[:, 5].astype(float))
        ws_bars.append(ws)

        # get fairness plot
        fairness = gmean(res_mps[:, 7].astype(float))
        fairness_bars.append(fairness)

    plot_mps(ws_bars, title='Weighted Speedup', ylabel='Weighted Speedup', ymax=2.5, 
            xticks=xticks, ideal=2)
    plot_mps(fairness_bars, title='Fairness', ylabel='Fairness', ymax=1.5, 
            xticks=xticks, ideal=1)


if __name__ == "__main__":
        main()
 
