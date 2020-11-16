import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
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
perfect_reached = int((num_jobs - violation.count)/num_jobs * 100)
special_entry = {'buffer': 'cached',
                 'count': gpu_count,
                 'qos_reached': perfect_reached,
                 'avg_err': violation.mean_error_pct(),
                 'stp': ws_total
                 }
data.append(special_entry)

df_data = pd.DataFrame(data)
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# f.set_size_inches(6, 8)

# GPU Count plot
# sns.barplot(x='buffer', y='count', ax=ax1, data=df_data)
# ax1.set_ylim([0, 80])
# ax1.set_ylabel('Number of GPUs Used')
#
# sns.barplot(x='buffer', y='qos_reached', ax=ax2, data=df_data)
# ax2.set_ylabel('QoS_reached')
#
# ax2_twin = ax2.twinx()
# sns.lineplot(x='buffer', y='avg_err', ax=ax2_twin, data=df_data, marker='o')
# ax2_twin.set_ylabel('Average Relative QoS Error (%)')
#
# sns.barplot(x='buffer', y='stp', ax=ax3, data=df_data)
# ax3.set_ylim([0, 2])
# ax3.set_ylabel('STP')
#
# plt.savefig('buffer.pdf', bbox_inches='tight')
# plt.close()

table = df_data.to_latex(
    columns=['buffer', 'count', 'qos_reached', 'avg_err', 'stp'],
    header=['Buffer', 'Num of GPUs', 'QoS_reached',
            'Average QoS Error (%)', 'STP'],
    index=False,
    bold_rows=True,
    float_format="%.2f"
)

print(table)
