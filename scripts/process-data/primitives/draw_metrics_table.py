#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
import os
from textwrap import wrap

# | DP Util/effic | SP Util/effic | HP Util/effic | DRAM Util | 
# DRAM read thruput | DRAM write thruput | L1/tex hit rate  | 
# L2 hit rate | Shared memory util | Special func unit util | 
# tensor FP util | tensor int8 util |

metrics = ['double_precision_fu_utilization',
          'flop_dp_efficiency',
          'single_precision_fu_utilization',
          'flop_sp_efficiency',
          'half_precision_fu_utilization',
          'flop_hp_efficiency',
          'dram_utilization',
          'dram_read_throughput',
          'dram_write_throughput',
          'global_hit_rate',
          'l2_tex_hit_rate',
          'shared_utilization',
          'special_fu_utilization',
          'tensor_precision_fu_utilization',
          'tensor_int_fu_utilization',
          'sm_efficiency']

def parse_run(filename, executable, output):
    # read data for isolated run
    data = pd.read_csv(filename, delimiter=',', 
                           skiprows=5, keep_default_na=False)

    data = data.drop(['Metric Description', 'Device'], 
                         axis=1)
    
    #create unique list of kernels
    KernelNames = data.Kernel.unique()
    
    table = []

    for k in KernelNames:
        k_df = data[data.Kernel == k]
        
        row = []
        
        row.append(executable)
        row.append(k[0 : k.find('(')])
        
        invocations = k_df['Invocations'].values[0]
        row.append(invocations)
     
        k_data = k_df.set_index('Metric Name').T.to_dict('list')
        
        for m in metrics:
            if m in k_data:
                value = k_data[m][-1]
                if '%' in value: 
                    value = value.replace('%', '')
                    value = '%.2f' % float(value) + '%'
                elif 'GB/s' in value:
                    value = value.replace('GB/s', '')
                    value = '%.2f' % float(value) + 'GB/s'
                elif 'MB/s' in value:
                    value = value.replace('MB/s', '')
                    value = '%.2f' % float(value) + 'MB/s'

                row.append(value)
            else:
                row.append('N/A')

        table.append(row)

    # normalize invocation weights first (column 2)
    table = np.array(table)
    a = table[:,2].astype(np.float)
    table[:, 2] = a / np.sum(a)
    tmp = []
    for x in table[:,2]:
        tmp.append('%.2f' % float(x))
    table[:,2] = np.array(tmp)
    table = table.tolist()
    
    # Print table into file...
    with open(output, 'a+') as f:
        for r in table:
            
            f.write(' | ')
            # remove me!
            f.write('Parboil | ')
            f.write(' | '.join(r))
            f.write(' | ')
        
            f.write('\n')
    
    
def parse_test(filepath, exec1, exec2, output):
    parse_run(os.path.join(filepath, 'single-1.csv'), exec1, output)
    parse_run(os.path.join(filepath, 'single-2.csv'), exec2, output)

def main():
    if (len(sys.argv) < 3):
         print("Usage: <path/to/script> <path/to/config> <path/to/output>")
         sys.exit()

    filepath = sys.argv[1]
    output = sys.argv[2]

    # parse config file and find the experiement result paths
    tests = pd.read_csv(filepath, delimiter=',', skiprows=0, keep_default_na=False).values

    if os.path.exists(output):
        os.remove(output)

    # testcase_no, device_name, exec1_name, exec2_name, keyword_exec1, keyword_exec2
    for test in tests:
        # parse the runs
        subdir = os.path.join('experiments',str(test[0]),'run0')
        parse_test(subdir, test[2], test[3], output)

if __name__ == "__main__":
        main()
