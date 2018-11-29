#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
import os.path

launch_overhead = {'sgemm': 11.7, 'compute_gemm': 64.5}

plt.rcParams["figure.figsize"] = (10,10)

def cut_tail(file1, file2, k_data1, k_data2):
    print('reading file... ', file1)
    print('reading file...', file2)

    # find out the length of the tail
    with open(file1) as f1:
        for line in f1:
            if "Total elapsed" in line:
                f1_elapsed = float(line.split(':')[1].strip().split(' ')[0])

    with open(file2) as f2:
        for line in f2:
            if "Total elapsed" in line:
                f2_elapsed = float(line.split(':')[1].strip().split(' ')[0])

    if (f1_elapsed > f2_elapsed):
        # cut tail of time-1 run
        offset = f1_elapsed - f2_elapsed
        checkpoint = k_data1[-1, 0] + k_data1[-1, 1] - offset
        k_data1 = k_data1[k_data1[:, 0] < checkpoint - 500]

        k_data1 = k_data1[200:]
        k_data2 = k_data2[200:]


    else:
        # cut tail of time-2 run
        offset = f2_elapsed - f1_elapsed
        checkpoint = k_data2[-1, 0] + k_data2[-1, 1] - offset
        k_data2 = k_data2[k_data2[:, 0] < checkpoint - 500]

        k_data1 = k_data1[200:]
        k_data2 = k_data2[200:]


    return k_data1, k_data2

# parse run() will return weighted speedup and unfairness of isolated runs, time sliced runs,
# and mps runs
# Assume only two concurrent applications
# Params: 
#       filepath: path to nvprof csv files. 
#               Naming convention is single-1/2.csv, time-1/2.csv, mps-1/2.csv
#       kw1/2: keyword to grab for the kernel names        

def parse_run(filepath, kw1, kw2):
    # read data for isolated run
    data_s_1 = pd.read_csv(filepath + '/single-1.csv', delimiter=',', skiprows=[0,1,2,4], keep_default_na=False)
    k_data_s_1 = data_s_1[data_s_1['Name'].str.contains(kw1)].values
    k_data_s_1 = k_data_s_1[1:, :2].astype(float); # drop the first run


    data_s_2 = pd.read_csv(filepath + '/single-2.csv', delimiter=',', skiprows=[0,1,2,4], keep_default_na=False)
    k_data_s_2 = data_s_2[data_s_2['Name'].str.contains(kw2)].values
    k_data_s_2 = k_data_s_2[1:, :2].astype(float); # drop the first run

    # read data for time sliced runs
    data_d_1 = pd.read_csv(filepath + '/time-1.csv', delimiter=',', skiprows=[0,1,2,4])
    k_data_d_1 = data_d_1[data_d_1['Name'].str.contains(kw1)].values
    k_data_d_1 = k_data_d_1[1:-1, :2].astype(float); # drop the first and last run

    data_d_2 = pd.read_csv(filepath + '/time-2.csv', delimiter=',', skiprows=[0,1,2,4])
    k_data_d_2 = data_d_2[data_d_2['Name'].str.contains(kw2)].values
    k_data_d_2 = k_data_d_2[1:-1, :2].astype(float); # drop the first and last run

    k_data_d_1, k_data_d_2 = cut_tail(filepath + '/time-1.txt', 
                                      filepath + '/time-2.txt',
                                      k_data_d_1,
                                      k_data_d_2)

    # read data for mps runs
    data_m_1 = pd.read_csv(filepath + '/mps-1.csv', delimiter=',', skiprows=[0,1,2,4])
    k_data_m_1 = data_m_1[data_m_1['Name'].str.contains(kw1)].values
    k_data_m_1 = k_data_m_1[1:-1, :2].astype(float); # drop the first and last run

    data_m_2 = pd.read_csv(filepath + '/mps-2.csv', delimiter=',', skiprows=[0,1,2,4])
    k_data_m_2 = data_m_2[data_m_2['Name'].str.contains(kw2)].values
    k_data_m_2 = k_data_m_2[1:-1, :2].astype(float); # drop the first and last run

    k_data_m_1, k_data_m_2 = cut_tail(filepath + '/mps-1.txt',
                                      filepath + '/mps-2.txt',
                                      k_data_m_1,
                                      k_data_m_2)

    ipc_s1 = k_data_s_1.shape[0] / (k_data_s_1[-1, 0] + k_data_s_1[-1,1] - k_data_s_1[0, 0])
    ipc_s2 = k_data_s_2.shape[0] / (k_data_s_2[-1, 0] + k_data_s_2[-1,1] - k_data_s_2[0, 0])
    ipc_t1 = k_data_d_1.shape[0] / (k_data_d_1[-1, 0] + k_data_d_1[-1,1] - k_data_d_1[0, 0])
    ipc_t2 = k_data_d_2.shape[0] / (k_data_d_2[-1, 0] + k_data_d_2[-1,1] - k_data_d_2[0, 0])
    ipc_m1 = k_data_m_1.shape[0] / (k_data_m_1[-1, 0] + k_data_m_1[-1,1] - k_data_m_1[0, 0])
    ipc_m2 = k_data_m_2.shape[0] / (k_data_m_2[-1, 0] + k_data_m_2[-1,1] - k_data_m_2[0, 0])

    slowd_t1 = ipc_t1 / ipc_s1
    slowd_t2 = ipc_t2 / ipc_s2
    slowd_m1 = ipc_m1 / ipc_s1
    slowd_m2 = ipc_m2 / ipc_s2

    ws_time = slowd_t1 + slowd_t2
    print('ws: ', ws_time)
    ws_mps = slowd_m1 + slowd_m2

    unfairness_time = max(slowd_t1, slowd_t2) / min(slowd_t1, slowd_t2)
    unfairness_mps = max(slowd_m1, slowd_m2) / min(slowd_m1, slowd_m2)
    print('unfairness: ', unfairness_time)

    return [ws_time, ws_mps, unfairness_time, unfairness_mps]

def reject_outliers(data, m = 3):
    data = data[abs(data - np.average(data)) < m * np.std(data)]
    return np.average(data), np.std(data)

def main():
    if (len(sys.argv) < 2):
        print("Usage: <path/to/script> <path/to/config>")
        sys.exit()

    filepath = sys.argv[1]

    # parse config file and find the experiement result paths
    tests = pd.read_csv(filepath, delimiter=',', skiprows=0, keep_default_na=False).values
    print(tests)

    ws = []
    unfair = []

    # testcase_no, device_name, exec1_name, exec2_name, keyword_exec1, keyword_exec2
    for test in tests:
        # parse the runs
        subdirs = [x[0] for x in os.walk('experiments/'+str(test[0]))][1:]
        data = [parse_run(x, test[-2], test[-1]) for x in subdirs]
        data = np.array(data)

        # store ws and unfairness in 
        # [ws_time,ws_time_err,ws_mps,ws_mps_err]
        # [unfair_time, unfair_time_err, unfair_mps, unfair_mps_err, app1,app2]

        avg_ws_t, st_ws_t = reject_outliers(data[:, 0])

        avg_ws_m, st_ws_m = reject_outliers(data[:, 1])

        print("WS")
        print(avg_ws_t,st_ws_t)
        print(avg_ws_m,st_ws_m)

        ws.append([avg_ws_t,st_ws_t,avg_ws_m,st_ws_m])

        avg_uf_t, st_uf_t = reject_outliers(data[:,2])

        avg_uf_m, st_uf_m = reject_outliers(data[:,3])

        print("unfairness")
        print(avg_uf_t,st_uf_t)
        print(avg_uf_m,st_uf_m)

        unfair.append([avg_uf_t,st_uf_t,avg_uf_m,st_uf_m])

    ws = np.array(ws)
    unfair = np.array(unfair)

    # set width of bar
    barWidth = 0.25
     
    # set height of bar
    bars1 = ws[:, 0]
    bars2 = ws[:, 2]
      
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
       
    # Make the plot
    plt.bar(r1, bars1, yerr = ws[:,1], color='#7f6d5f', width=barWidth, edgecolor='white', label='Time Sliced')
    plt.bar(r2, bars2, yerr = ws[:,3], color='#557f2d', width=barWidth, edgecolor='white', label='MPS')
        
    # Add xticks on the middle of the group bars
    plt.xlabel('Test Cases', fontweight='bold')
    legends = [ test[2] + '+' + test[3] for test in tests]
    plt.xticks([r + barWidth/2 for r in range(len(bars1))], legends)
         
    # Create legend & Show graphic
    plt.legend()

    plt.title(tests[0, 1] + " Weighted Speedup")
    plt.ylabel('Weighted speedup')
    plt.hlines(2, -0.5, 1.5, linestyles='dashed', label='ideal')
    plt.savefig('experiments/weighted_speedup.png')
    plt.close()

    # set height of bar
    bars1 = unfair[:, 0]
    bars2 = unfair[:, 2]
      
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
       
    # Make the plot
    plt.bar(r1, bars1, yerr = unfair[:,1], color='#7f6d5f', width=barWidth, edgecolor='white', label='Time Sliced')
    plt.bar(r2, bars2, yerr = unfair[:,3], color='#557f2d', width=barWidth, edgecolor='white', label='MPS')
        
    # Add xticks on the middle of the group bars
    plt.xlabel('Test Cases', fontweight='bold')
    legends = [ test[2] + '+' + test[3] for test in tests]
    plt.xticks([r + barWidth/2 for r in range(len(bars1))], legends)
         
    # Create legend & Show graphic
    plt.legend()

    plt.title(tests[0, 1] + " Unfairness")
    plt.ylabel('Unfairness')
    plt.hlines(1, -0.5, 1.5, linestyles='dashed', label='ideal')
    plt.savefig('experiments/unfairness.png')
    plt.close()













    #plt.bar([0,1], [avg_ws_t, avg_ws_m], yerr=[st_ws_t, st_ws_m], width=0.5)
    #plt.xticks([0,1], ['time', 'mps'])
    #plt.title(title + " Weighted Speedup")
    #plt.ylabel('Weighted speedup')
    #plt.hlines(2, -0.5, 1.5, linestyles='dashed', label='ideal')
    #plt.savefig(filepath + '/weighted_speedup.png')
    #plt.close()

    #plt.bar([0,1], [avg_uf_t, avg_uf_m], yerr=[st_uf_t, st_uf_m], width=0.5)
    #plt.xticks([0,1], ['time', 'mps'])
    #plt.title(title + ' Unfairness')
    #plt.ylabel('Unfairness')
    #plt.hlines(1, -0.5, 1.5, linestyles='dashed', label='ideal')
    #plt.savefig(filepath + '/unfairness.png')
    #plt.close()





if __name__ == "__main__":
        main()
