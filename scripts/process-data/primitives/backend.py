import numpy as np
import pandas as pd
import math
import os.path

def read_runtime(filename):

    data = pd.read_csv(filename, delimiter=',', 
                       skiprows=[0,1,2,4], keep_default_na=False)

    return data


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
 
# Return a list of IPCs in the order of incoming apps
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

        ipc.append(ipc_app)
        
    return ipc


def cut_tail(file1, file2, data1, data2):
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

    else:
        # cut tail of time-2 run
        offset = f2_elapsed - f1_elapsed
        checkpoint = k_data2[-1, 0] + k_data2[-1, 1] - offset
        k_data2 = k_data2[k_data2[:, 0] < checkpoint]
        #print(k_data2.shape)

    df_1 = pd.DataFrame(data=k_data1, columns=data1.columns.values)
    df_2 = pd.DataFrame(data=k_data2, columns=data2.columns.values)
    
#     print(df_1)
#     print(df_2)
    
    return df_1, df_2

# Calculate app weighted throughput, system weighted speedup, system fairness
def calculate_sys_metrics(ipc_s, ipc_c):
    weighted_thruput = []
    ws = 0
    fairness = 0 

    slowd_arr = []

    for i in range(len(ipc_s)):
        slowd = ipc_c[i] / ipc_s[i]
        weighted_thruput.append(slowd)

        ws = ws + slowd
        slowd_arr.append(slowd)
        
    fairness = min (slowd_arr[0]/slowd_arr[1], slowd_arr[1]/slowd_arr[0])

    return weighted_thruput + [ws, fairness]


def parse_run(filepath, inst_maps, time, mps):
    results = {}

    # read data for isolated run
    data_s = []
    data_s.append(read_runtime(filepath + '/single-1.csv'))
    data_s.append(read_runtime(filepath + '/single-2.csv'))
    
    # calculate ipc of isolated runs 
    ipc_s = calculate_ipc_apps(data_s, inst_maps)

    if (time == 1):
        # process time multiplexing data

        data_t = []
        data_t.append(read_runtime(filepath + '/time-1.csv'))
        data_t.append(read_runtime(filepath + '/time-2.csv'))
        
        data_t[0], data_t[1] = cut_tail(filepath + '/time-1.txt',
                                          filepath + '/time-2.txt',
                                          data_t[0],
                                          data_t[1])
        

        ipc_t = calculate_ipc_apps(data_t, inst_maps)
        sys_t = calculate_sys_metrics(ipc_s, ipc_t)

        results['time'] = sys_t
        

    if (mps == 1):
        # process data for mps runs

        data_m = []
        data_m.append(read_runtime(filepath + '/mps-1.csv')) 
        data_m.append(read_runtime(filepath + '/mps-2.csv')) 
        data_m[0], data_m[1] = cut_tail(filepath + '/mps-1.txt',
                                          filepath + '/mps-2.txt',
                                          data_m[0],
                                          data_m[1])
        

        ipc_m = calculate_ipc_apps(data_m, inst_maps)
        sys_m = calculate_sys_metrics(ipc_s, ipc_m)
        
        results['mps'] = sys_m

    return results

def reject_outliers(data, m = 3):
    data = data[abs(data - np.average(data)) < m * np.std(data)]
    return np.average(data), np.std(data)



