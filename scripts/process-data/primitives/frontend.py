from .backend import * 

import numpy as np
import pandas as pd
import math
import os.path


   
def get_run_inst(filepath):
    maps = []
    maps.append(parse_app_inst(os.path.join(filepath, 'single-1_inst.csv')))
    maps.append(parse_app_inst(os.path.join(filepath, 'single-2_inst.csv')))
    
    return maps
 

def print_sys_perf(configname, resultpath):
    # parse config file and find the experiement result paths
    # testcase_no, device_name, exec1_name, exec2_name, keyword_exec1, keyword_exec2, time, mps
    tests = pd.read_csv(configname, delimiter=',', keep_default_na=False).values
    print(tests)

    ws = []
    fair = []
    # results = [test No., time-1/mps-0, 
    # (wt_avg, wt_std, ws_avg, ws_std, fair_avg, fair_std) x execs]
    res_time = []
    res_mps = []
    
    # Loop through all tests in the config file
    for test in tests:

        # first, get the instruction counts per kernel
        inst_maps = get_run_inst(os.path.join(resultpath, test[0], 'run0'))

        # parse the runs
        subdirs = [x[0] for x in os.walk(os.path.join(resultpath, test[0]))][1:]
        # data will store a list of {time: [], mps: []} from each run
        data = [parse_run(run, inst_maps, time=test[-2], mps=test[-1]) for run in subdirs]

        # process time multiplexing data
        if (test[-2] == 1):
            result_test = [test[0]]

            data_t = [d['time'] for d in data]
            data_t = np.array(data_t)

            for col in data_t.T:
                avg, std = reject_outliers(col)
                result_test = result_test + [avg,std]

            res_time.append(result_test)

        # process mps data
        if (test[-1] == 1):
            result_test = [test[0]]

            data_m = [d['mps'] for d in data]
            data_m = np.array(data_m)

            for col in data_m.T:
                avg, std = reject_outliers(col)
                result_test = result_test + [avg,std]

            res_mps.append(result_test)

    # print results
    print('Table: time multiplexing')
    for row in res_time:
        print(row)

    print('Table: mps')
    for row in res_mps:
        print(row)


    return res_time, res_mps



