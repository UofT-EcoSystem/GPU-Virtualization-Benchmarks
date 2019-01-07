#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
import os.path
from textwrap import wrap


plt.rcParams["figure.figsize"] = (15,10)

def cut_tail(file1, file2, data1, data2):
    #print('reading file... ', file1)
    #print('reading file...', file2)
    
    k_data1 = data1.values
    k_data2 = data2.values

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
        k_data1 = k_data1[k_data1[:, 0] < checkpoint]

#         k_data1 = k_data1[200:]
#         k_data2 = k_data2[200:]


    else:
        # cut tail of time-2 run
        offset = f2_elapsed - f1_elapsed
        checkpoint = k_data2[-1, 0] + k_data2[-1, 1] - offset
        k_data2 = k_data2[k_data2[:, 0] < checkpoint]
        #print(k_data2.shape)

#         k_data1 = k_data1[200:]
#         k_data2 = k_data2[200:]


    df_1 = pd.DataFrame(data=k_data1, columns=data1.columns.values)
    df_2 = pd.DataFrame(data=k_data2, columns=data2.columns.values)
    
#     print(df_1)
#     print(df_2)
    
    return df_1, df_2

def read_runtime(filename):

    data = pd.read_csv(filename, delimiter=',', 
                       skiprows=[0,1,2,4], keep_default_na=False)

#     data = data[:, :2].astype(float)

    return data

def calculate_ipc_apps(arr_data, inst_maps):
    # IPC = sum(#weight * #inst * #invoks / T)
        
    ipc = []
    for i in range(len(arr_data)):
        ipc_app = 0
        data = arr_data[i]
        inst_dict = inst_maps[i]

        T = data.iloc[-1].Duration + data.iloc[-1].Start - data.iloc[0].Start
        
        for kernel in inst_dict:
            str_kernel = kernel[0: kernel.find('(')]
            #print(str_kernel)
            #print('old:', str(len(data)))

            k_data = data[data.Name.str.contains(str_kernel)]

            n_invoks = len(k_data)
            #print(n_invoks)

            #T = k_data['Duration'].sum()

            #k_ipc = inst_dict[kernel][0] * inst_dict[kernel][1] * n_invoks / T
            k_ipc = inst_dict[kernel][1] * n_invoks

            ipc_app = ipc_app + k_ipc

        # end for kernel
        ipc_app = ipc_app / float(T)
        print(ipc_app)

        ipc.append(ipc_app)
        
    return ipc

# parse run() will return weighted speedup and unfairness of isolated runs, time sliced runs,
# and mps runs
# Assume only two concurrent applications
# Params: 
#       filepath: path to nvprof csv files. 
#               Naming convention is single-1/2.csv, time-1/2.csv, mps-1/2.csv
#       kw1/2: keyword to grab for the kernel names        

def parse_run(filepath, inst_maps):
    print(filepath)
    # read data for isolated run
    data_s = []
    data_s.append(read_runtime(filepath + '/single-1.csv'))
    data_s.append(read_runtime(filepath + '/single-2.csv'))
    
    # calculate ipc of isolated runs
    ipc_s = calculate_ipc_apps(data_s, inst_maps)


    # read data for time sliced runs
    data_d = []
    data_d.append(read_runtime(filepath + '/time-1.csv')) 
    data_d.append(read_runtime(filepath + '/time-2.csv'))
    
    # FIXME: lazy use of cuttail
    # need to expand this to an arbitrary number of applications
    data_d[0], data_d[1] = cut_tail(filepath + '/time-1.txt', 
                                      filepath + '/time-2.txt',
                                      data_d[0],
                                      data_d[1])

    
    # calculate ipc of time sliced runs
    ipc_d = calculate_ipc_apps(data_d, inst_maps)
    
    # calculate ws and unfairness of time sliced runs
    ws_time = 0
    uf_time = 0 
    uf_time_idx = 0
    slowd_arr = []
    for i in range(len(data_d)):
        #print('ipc_s:',ipc_s[i])
        #print('ipc_d:',ipc_d[i])
        slowd = ipc_d[i] / ipc_s[i]
        ws_time = ws_time + slowd
        slowd_arr.append(slowd)
     
    uf_time = min (slowd_arr[0]/slowd_arr[1], slowd_arr[1]/slowd_arr[0])

    # read data for mps runs
    data_m = []
    data_m.append(read_runtime(filepath + '/mps-1.csv'))
    data_m.append(read_runtime(filepath + '/mps-2.csv'))
    
    data_m[0], data_m[1] = cut_tail(filepath + '/mps-1.txt',
                                      filepath + '/mps-2.txt',
                                      data_m[0],
                                      data_m[1])

    # calculate ipc of mps runs
    ipc_m = calculate_ipc_apps(data_m, inst_maps)
    
    # calculate ws and unfairness of mps runs
    ws_mps = 0
    uf_mps = 0 
    uf_mps_idx = 0
    slowd_arr = []
    for i in range(len(data_m)):
        slowd = ipc_m[i] / ipc_s[i]
        print('slowd:', slowd)
        ws_mps = ws_mps + slowd
        slowd_arr.append(slowd)
     

    uf_mps = min (slowd_arr[0]/slowd_arr[1], slowd_arr[1]/slowd_arr[0])
    if (slowd_arr[0]/slowd_arr[1] < slowd_arr[1]/slowd_arr[0]):
        print('0 is slow')

    return [ws_time, ws_mps, uf_time, uf_mps, uf_time_idx, uf_mps_idx]

def reject_outliers(data, m = 3):
    data = data[abs(data - np.average(data)) < m * np.std(data)]
    return np.average(data), np.std(data)

def parse_app_inst(filename):

    data = pd.read_csv(filename, delimiter=',',
               skiprows=4, keep_default_na=False)
    
    data = data.drop(['Metric Description', 'Device', 'Metric Name', 'Min', 'Max'],
                         axis=1)
    
    invoks_sum = data['Invocations'].sum()
    data['Invocations'] = data['Invocations'] / invoks_sum
    
    # Now make data into a dictionary
    # {kernel: [weight, #inst]}
    data = data.set_index('Kernel').T.to_dict('list')

    
    return data
    
def get_run_inst(filepath):
    maps = []
    #maps.append(parse_app_inst(os.path.join(filepath, 'single-1_inst.csv')))
    #maps.append(parse_app_inst(os.path.join(filepath, 'single-2_inst.csv')))
    maps.append({'spmv':[1, 100]})
    maps.append({'gemm':[1, 100]})
    
    return maps
    
    
    
def main():
    if (len(sys.argv) < 3):
         print("Usage: <path/to/script> <path/to/config> <plot name>")
         sys.exit()

    filepath = sys.argv[1]
    plotname = sys.argv[2]

    # parse config file and find the experiement result paths
    tests = pd.read_csv(filepath, delimiter=',', skiprows=0, keep_default_na=False).values
    print(tests)

    ws = []
    unfair = []
    

    # testcase_no, device_name, exec1_name, exec2_name, keyword_exec1, keyword_exec2
    for test in tests:
        print(test)
        # first, get the instruction counts
        inst_maps = get_run_inst(os.path.join('experiments',test[0], 'run0'))

            
        # parse the runs
        subdirs = [x[0] for x in os.walk('experiments/'+str(test[0]))][1:]
        data = [parse_run(x, inst_maps) for x in subdirs]
        data = np.array(data)

        # store ws and unfairness in 
        # [ws_time,ws_time_err,ws_mps,ws_mps_err]
        # [unfair_time, unfair_time_err, unfair_mps, unfair_mps_err, app1,app2]

        avg_ws_t, st_ws_t = reject_outliers(data[:, 0])

        avg_ws_m, st_ws_m = reject_outliers(data[:, 1])

        #print("WS")
        #print(avg_ws_t,st_ws_t)
        #print(avg_ws_m,st_ws_m)

        ws.append([avg_ws_t,st_ws_t,avg_ws_m,st_ws_m])

        avg_uf_t, st_uf_t = reject_outliers(data[:,2])

        avg_uf_m, st_uf_m = reject_outliers(data[:,3])

        #print("unfairness")
        #print(avg_uf_t,st_uf_t)
        #print(avg_uf_m,st_uf_m)

        unfair.append([avg_uf_t,st_uf_t,avg_uf_m,st_uf_m,np.average(data[:,4]), np.average(data[:,5])])

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
    plt.bar(r1, bars1, yerr = ws[:,1], color='#7f6d5f', width=barWidth, edgecolor='white', label='Time Sliced', capsize=3)
    plt.bar(r2, bars2, yerr = ws[:,3], color='#557f2d', width=barWidth, edgecolor='white', label='MPS', capsize=3)
        
    # Add xticks on the middle of the group bars
    plt.xlabel('Test Cases', fontsize=20)
    legends = [ test[2] + '\n' + test[3] for test in tests]
    legends_tex = ['{256}','{512}', '{1024}', '{2048}', '{2048 (1:1)}']
    plt.xticks([r + barWidth/2 for r in range(len(bars1))], legends)
         
    # Create legend & Show graphic
    plt.legend()

    plt.title(tests[0, 1] + " Weighted Speedup", fontsize=24)
    plt.ylabel('Weighted speedup', fontsize=20)
    plt.axhline(2, zorder=3, color='black', linestyle='--')
    plt.text(len(bars2)+barWidth*2,1.99,'ideal', fontsize=20)
    plt.ylim([0,2.5])
    plt.savefig('experiments/' + plotname + '_weighted_speedup.pdf', transparent=True)
    plt.close()


    print('WS-Time')
    for i in range(len(bars1)):
        print('(', legends_tex[i], ' , ', bars1[i], ')+-(', ws[:,1][i], ',', ws[:,1][i], ')')

    print('WS-MPS')
    for i in range(len(bars2)):
        print('(', legends_tex[i], ' , ', bars2[i], ')+-(', ws[:,3][i], ',', ws[:,3][i], ')')

 

    # set height of bar
    bars1 = unfair[:, 0]
    bars2 = unfair[:, 2]
      
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
       
    # Make the plot
    plt.bar(r1, bars1, yerr = unfair[:,1], color='#7f6d5f', width=barWidth, edgecolor='white', label='Time Sliced', capsize=3)
    #plt.text(r1, bars1, str(unfair[:, 4]))
    #print(unfair[:, 4])
    plt.bar(r2, bars2, yerr = unfair[:,3], color='#557f2d', width=barWidth, edgecolor='white', label='MPS', capsize=3)
    #plt.text(r2, bars2, str(unfair[:, 5]))
    #print(unfair[:, 4])

       
    # Add xticks on the middle of the group bars
    plt.xlabel('Test Cases', fontsize = 20)
    legends = [ test[2] + '\n' + test[3] for test in tests]
    plt.ylim([0, math.ceil(max(np.max(bars1 + unfair[:,1]), np.max(bars2 + unfair[:,3])))])
    plt.xticks([r + barWidth/2 for r in range(len(bars1))], legends)
         
    # Create legend & Show graphic
    plt.legend()

    plt.title(tests[0, 1] + " Unfairness", fontsize=24)
    plt.ylabel('Unfairness', fontsize=20)
    plt.axhline(1, zorder=3, color='black', linestyle='--')
    plt.text(len(bars2)+barWidth*2,0.99,'ideal', fontsize=20)
    plt.savefig('experiments/' + plotname + '_unfairness.pdf', transparent=True)
    plt.close()

    
    print('UF-Time')
    for i in range(len(bars1)):
        print('(', legends_tex[i], ' , ', bars1[i], ')')
        print('(', legends_tex[i], ' , ', bars1[i], ')+-(', unfair[:,1][i], ',', unfair[:,1][i], ')')

    print('UF-MPS')
    for i in range(len(bars2)):
        print('(', legends_tex[i], ' , ', bars2[i], ')')
        print('(', legends_tex[i], ' , ', bars2[i], ')+-(', unfair[:,3][i], ',', unfair[:,3][i], ')')

   



if __name__ == "__main__":
        main()
        
