import pandas as pd
import numpy as np
import sys
from gpupool.predict import PairJob
from gpupool.workload import BatchJob, GpuPoolConfig
from gpupool.predict import Allocation, StageOne, StageTwo
from gpupool.workload import Violation

num_jobs = 100
batch = BatchJob(rand_seed=0, num_jobs=num_jobs)
batch.load_df_from_pickle(
    "pickles/eco12/BatchJob-0-Three_D-BoostTree-Steady-False.pkl")

buffers = np.arange(0, 0.12, 0.02)

data = []
for buffer in buffers:
    config = GpuPoolConfig(Allocation.Three_D, StageOne.BoostTree,
                           StageTwo.Steady,
                           at_least_once=False, accuracy_mode=False,
                           stage2_buffer=buffer)

    print("Buffer = ", buffer)
    gpupool_count, gpupool_violation, gpupool_ws, ws_list_gpupool, isolated = \
        batch.max_matching(config, cores=32)
    print("GPUPool GPU count:", gpupool_count)
    print("WS:", gpupool_ws)

    gpu_count_no_migrate = gpupool_count - gpupool_violation.gpu_increase
    ws_no_migrate = (gpupool_violation.ws_no_migrate + isolated) / \
                    gpu_count_no_migrate
    print("gpu count no migrate:", gpu_count_no_migrate)
    print("WS no migrate:", ws_no_migrate)
    print(gpupool_violation.to_string(num_jobs))
    print("\n")

    qos_reached = int((num_jobs - gpupool_violation.count)/num_jobs * 100)
    entry = {'buffer': "{}%".format(int(buffer * 100)),
             'count': gpu_count_no_migrate,
             'qos_reached': qos_reached,
             'avg_err': gpupool_violation.mean_error_pct(),
             'stp': ws_no_migrate
             }
    data.append(entry)

# Perfect stage 1:
config = GpuPoolConfig(Allocation.Three_D, StageOne.GPUSim,
                       StageTwo.Steady,
                       at_least_once=False, accuracy_mode=False,
                       stage2_buffer=0)
gpu_count, violation, ws_total = batch.calculate_gpu_count_gpupool(
    config, 32, save=False)
print("Perfect stage 1")
print("GPU count:", gpu_count)
print("violations: ", violation.count)
print("ws_total", ws_total)
perfect_reached = int((num_jobs - violation.count) / num_jobs * 100)
special_entry = {'buffer': 'cached',
                 'count': gpu_count,
                 'qos_reached': perfect_reached,
                 'avg_err': violation.mean_error_pct(),
                 'stp': ws_total
                 }

# debugging

# bad_pair = PairJob([batch.list_jobs[40], batch.list_jobs[80]])
# result = bad_pair.get_gpupool_performance(config)
# option_col = config.get_option()
# result[option_col].get_real_performance()

data.append(special_entry)

df_data = pd.DataFrame(data)

table = df_data.to_latex(
    columns=['buffer', 'count', 'qos_reached', 'avg_err', 'stp'],
    header=['Buffer', 'Num of GPUs', 'QoS_reached',
            'Average QoS Error (%)', 'STP'],
    index=False,
    bold_rows=True,
    float_format="%.2f"
)

print(table)
