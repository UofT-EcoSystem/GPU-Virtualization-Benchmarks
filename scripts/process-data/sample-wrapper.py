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


def main():
    if (len(sys.argv) < 2):
         print("Usage: <path/to/script> <path/to/config>")
         sys.exit()

    config = sys.argv[1] 
    res_time, res_mps = print_sys_perf(config, 
                            os.path.join(os.path.dirname(os.path.abspath(config)), 'experiments'))



    # set width of bar
    barWidth = 0.25

    if (len(res_time) != 0 and len(res_mps) != 0):
        # draw grouped bars
        res_time = np.array(res_time)
        res_mps = np.array(res_mps)

        bars1 = res_time[:, 5].astype(float)
        err1 = res_time[:, 6].astype(float)

        bars2 = res_mps[:, 5].astype(float)
        err2 = res_mps[:, 6].astype(float)

        r1 = np.arange(len(bars1))
        r2 = [x + barWidth for x in r1]
     
        # Make the plot
        plt.bar(r1, bars1, yerr = err1, color='#7f6d5f', edgecolor='white', 
                width=barWidth, label='Time Sliced', capsize=3)
        plt.bar(r2, bars2, yerr = err2, color='#557f2d', edgecolor='white',
                width=barWidth, label='MPS', capsize=3)
 
        legends = res_time[:, 0] 
        plt.xticks([r + barWidth/2 for r in range(len(bars1))], legends, fontsize=18)

#    elif (len(res_time) != 0):
        # time multiplexing only
#    elif (len(res_mps) != 0):
        # mps only

    
    plt.legend(fontsize=20)
       
    # Add xticks on the middle of the group bars
    plt.xlabel('Test Cases', fontsize=20)
         
    plt.title(" Weighted Speedup", fontsize=24)
    plt.ylabel('Weighted speedup', fontsize=20)
    plt.yticks(fontsize=18)
    plt.axhline(2, zorder=3, color='black', linestyle='--')
    plt.ylim([0,2.5])
    #plt.savefig('weighted_speedup.png', transparent=False)
    #plt.close()

    tikz_save('sample.tex')
 
    '''
    print('gmean WS-MPS')
    print(gmean(bars1))

    # set height of bar
    bars1 = fair[:, 0]
      
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
       
    # Make the plot
    plt.bar(r1, bars1, yerr = fair[:,1], color='#7f6d5f', width=barWidth, edgecolor='white', label='Time Sliced', capsize=3)
    #plt.text(r1, bars1, str(fair[:, 4]))
    #print(fair[:, 4])
    #plt.text(r2, bars2, str(fair[:, 5]))
    #print(fair[:, 4])

       
    # Add xticks on the middle of the group bars
    plt.xlabel('Test Cases', fontsize = 20)
    legends = [ test[2] + '\n' + test[3] for test in tests]
    plt.xticks([r + barWidth/2 for r in range(len(bars1))], legends)
         
    # Create legend & Show graphic
    plt.legend()

    plt.title(tests[0, 1] + " Unfairness", fontsize=24)
    plt.ylabel('Unfairness', fontsize=20)
    plt.axhline(1, zorder=3, color='black', linestyle='--')
    plt.savefig('experiments/' + plotname + '_fairness.pdf', transparent=True)
    plt.close()

    print('UF-Time')
    for i in range(len(bars1)):
        print('(', legends_tex[i], ' , ', bars1[i], ')')

    print('UF-MPS')
    for i in range(len(bars2)):
        print('(', legends_tex[i], ' , ', bars2[i], ')')


    print('gmean UF-MPS')
    print(gmean(bars1))
    '''

if __name__ == "__main__":
        main()
 
