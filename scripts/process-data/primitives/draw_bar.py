import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import re
import difflib


# draw stacked bar charts (verticle)
def draw_stack(results, legends, title, xlabel, ylabel, xticks, xtick_rot=0, outfile='inst.pdf'):
    # draw stack bars
    idx = np.arange(results.shape[0])
    width = 0.35

    accum = np.zeros(results.shape[0], dtype=np.float32)

    p = []
    # colors = sns.color_palette("husl", len(cols)-1)
    colors = sns.color_palette("Set2", len(legends))

    for i, col in enumerate(results.T):
        if i == 0 :
            continue
        #FIXME: remove this, this logic should go to top level
        #height = np.divide(col, results[:,-2])
        p.append(plt.bar(idx, col, width, bottom=accum, color=colors[i-1]))
        accum += col.astype(np.float32)

    # add the other uncaptured types
    #p.append(plt.bar(idx, 1-accum, width, bottom=accum))

    #plt.xticks(idx, results[:, 0], fontsize=16, rotation=30)
    plt.xticks(idx, xticks, fontsize=16, rotation=xtick_rot)
    plt.yticks(fontsize=16)

    plt.title(title, fontsize=22)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    #plt.ylim([0, 1])
    # flip legends
    p.reverse()
    legends.reverse()

    plt.legend(p, legends, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., fontsize=18)
    # TODO: print kernels in xticks

    plt.savefig(outfile,  bbox_inches='tight')
    plt.close()

# draw normal single bar chart
def draw_bar(results, legends, title, xlabel, ylabel, xticks, xtick_rot=0, outfile='util.pdf'):
    num_metrics = results.shape[1]
    idx = np.arange(results.shape[0])

    p = []
    colors = sns.color_palette("husl", len(legends))
    #colors = sns.color_palette("Set2", len(legends))

    ax1 = None
    for i, col in enumerate(results.T):
        if i == 0:
            continue

        if i == 1:
            ax = ax1 = plt.subplot(len(legends), 1, i)
            plt.bar(idx, results[:, i], color=colors[i-1], label=legends[i-1])
            plt.title(title, fontsize=22)
        else:
            ax = plt.subplot(len(legends), 1, i, sharex=ax1, sharey=ax1)
            plt.bar(idx, results[:, i], color=colors[i-1], label=legends[i-1])


        plt.setp(ax.get_xticklabels(), rotation=20)
        #plt.setp(plt.xticks(), rotation=30)
        plt.legend(fontsize=18)
        plt.ylim([0, 100])
        plt.yticks(fontsize=16)


        if i == num_metrics//2:
            plt.ylabel(ylabel, fontsize=18)
        if i == num_metrics - 1:
            plt.xlabel(xlabel, fontsize=18)

    plt.xticks(idx, xticks, fontsize=14, rotation=xtick_rot)
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()



