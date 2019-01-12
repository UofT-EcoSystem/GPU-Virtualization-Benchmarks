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

def plot_grouped(bars1, err1, bars2, err2, xticks, title, ylabel, ymax, ideal):
    # set width of bar
    barWidth = 0.25

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
 
    # Make the plot
    plt.bar(r1, bars1, yerr = err1, color='#7f6d5f', edgecolor='white', 
            width=barWidth, label='Time Sliced', capsize=3)
    plt.bar(r2, bars2, yerr = err2, color='#557f2d', edgecolor='white',
            width=barWidth, label='MPS', capsize=3)

    plt.xticks([r + barWidth/2 for r in range(len(bars1))], xticks, fontsize=18)

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
 


def plot_mps(bars, err, xticks, title, ylabel, ymax, ideal):
    # set width of bar
    barWidth = 0.25

    r = np.arange(len(bars))

    plt.bar(r, bars, yerr = err, color='#7f6d5f', edgecolor='white', 
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
         print("Usage: <path/to/script> <path/to/config>")
         sys.exit()

    config = sys.argv[1] 
    res_time, res_mps = print_sys_perf(config, 
                            os.path.join(os.path.dirname(os.path.abspath(config)), 'experiments'))


    # Make plots:

    if (len(res_time) != 0 and len(res_mps) != 0):
        # draw grouped bars
        res_time = np.array(res_time)
        res_mps = np.array(res_mps)

        # make weighted speedup plot
        bars1 = res_time[:, 5].astype(float)
        err1 = res_time[:, 6].astype(float)

        bars2 = res_mps[:, 5].astype(float)
        err2 = res_mps[:, 6].astype(float)

        plot_grouped(bars1, err1, bars2, err2, xticks=res_time[:, 0], ideal=2,
                title='Weighted Speedup', ylabel='Weighted Speedup', ymax=2.5)
        
        # make fairness plot
        bars1 = res_time[:, 7].astype(float)
        err1 = res_time[:, 8].astype(float)

        bars2 = res_mps[:, 7].astype(float)
        err2 = res_mps[:, 8].astype(float)

        plot_grouped(bars1, err1, bars2, err2, xticks=res_time[:, 0], ideal=1,
                title='Fairness', ylabel='Fairness', ymax=1.5)
 

    elif (len(res_mps) != 0):
        # mps only
        res_mps = np.array(res_mps)

        # make weighted speedup plot
        bars = res_mps[:, 5].astype(float)
        err = res_mps[:, 6].astype(float)

        plot_mps(bars, err, xticks=res_mps[:, 0], ideal=2,
                title='Weighted Speedup', ylabel='Weighted Speedup', ymax=2.5)

        # make fairness plot
        bars = res_mps[:, 7].astype(float)
        err = res_mps[:, 8].astype(float)

        plot_mps(bars, err, xticks=res_mps[:, 0], ideal=1,
                title='Fairness', ylabel='Fairness', ymax=1.5)






if __name__ == "__main__":
        main()
 
