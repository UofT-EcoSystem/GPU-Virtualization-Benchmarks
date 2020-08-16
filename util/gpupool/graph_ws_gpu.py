import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from gpupool.workload import BatchJob, GpuPoolConfig
from gpupool.predict import Allocation, StageOne, StageTwo
from gpupool.workload import Violation

num_jobs = 100
batch = BatchJob(rand_seed=0, num_jobs=num_jobs)
batch.load_df_from_pickle(
    "pickles/BatchJob-0-Three_D-BoostTree-Steady-False.pkl")

# GPUPool
config = GpuPoolConfig(Allocation.Three_D, StageOne.BoostTree, StageTwo.Steady,
                       at_least_once=False, accuracy_mode=False,
                       stage2_buffer=0.1)

gpupool_count, gpupool_violation, gpupool_ws, ws_list_gpupool = \
    batch._max_matching(config, cores=32)
print("GPUPool GPU count:", gpupool_count)
print("WS:", gpupool_ws)
print(gpupool_violation.to_string(num_jobs))

# MIG
mig_count, mig_ws, ws_list_mig = batch.calculate_gpu_count_mig()
print("MIG count:", mig_count)
print("MIG WS", mig_ws)

# Heuristic
violations_count = num_jobs
heuristic_count = num_jobs / 2

heuristic_migrated_count_final = num_jobs
ws_list_heuristic_final = []
heuristic_count_final = num_jobs
heuristic_violation_final = Violation()
# Iteratively find the best config
while violations_count > 0:
    heuristic_count += 2
    violation, ws_list, gpu_migrated = \
        batch.calculate_qos_viol_dram_bw(heuristic_count,
                                         cores=32)
    violations_count = violation.count
    if gpu_migrated < heuristic_migrated_count_final:
        heuristic_migrated_count_final = gpu_migrated
        ws_list_heuristic_final = ws_list
        heuristic_count_final = heuristic_count
        heuristic_violation_final = violation

heuristic_ws_final = \
    sum(ws_list_heuristic_final) / heuristic_migrated_count_final
print("Heuristic count", heuristic_migrated_count_final, heuristic_count_final)
print("Heuristic WS", heuristic_ws_final)

sorted_gpupool_ws = sorted(ws_list_gpupool, reverse=True)
gpupool_xs = range(len(ws_list_gpupool))

sorted_mig_ws = sorted(ws_list_mig, reverse=True)
mig_xs = range(len(ws_list_mig))

sorted_heuristic_ws = sorted(ws_list_heuristic_final, reverse=True)
heuristic_xs = range(len(ws_list_heuristic_final))

# WS per GPU
# plt.style.use('seaborn-paper')
sns.set_style("ticks")

sns.lineplot(x=heuristic_xs, y=sorted_heuristic_ws, marker='<',
             label='Heuristic',
             zorder=10, clip_on=False)
sns.lineplot(x=mig_xs, y=sorted_mig_ws, marker='s', label='Coarse-Grained',
             zorder=11, clip_on=False)
sns.lineplot(x=gpupool_xs, y=sorted_gpupool_ws, clip_on=False,
             zorder=12, marker='.', label='GPUPool')
plt.xlabel("GPU")
plt.ylabel("STP")
plt.ylim([1, 2])

plt.savefig("ws_gpu.pdf", bbox_inches='tight')
plt.close()

# Draw bar graphs for comparison
plt.figure(figsize=(4, 4))
ax = sns.barplot(x=["No-Sharing", "Coarse-Grained", "Heuristic", "GPUPool"],
                 y=[num_jobs, mig_count, heuristic_migrated_count_final,
                    gpupool_count],
                 color='w', edgecolor='k'
                 )
plt.ylabel("Required GPU Count")
plt.xticks(rotation=30)
hatches = itertools.cycle(['//', '-', '+', 'x'])
for i, bar in enumerate(ax.patches):
    hatch = next(hatches)
    bar.set_hatch(hatch)

plt.savefig("required_gpu_count.pdf", bbox_inches='tight')
plt.close()

# Draw aggregate WS stats
plt.figure(figsize=(4, 4))
ax = sns.barplot(x=["No-Sharing", "Coarse-Grained", "Heuristic", "GPUPool"],
                 y=[1, mig_ws, heuristic_ws_final, gpupool_ws],
                 color='w', edgecolor='k'
                 )
plt.ylabel("STP")
plt.xticks(rotation=30)
hatches = itertools.cycle(['//', '-', '+', 'x'])
for i, bar in enumerate(ax.patches):
    hatch = next(hatches)
    bar.set_hatch(hatch)

plt.savefig("aggregate_ws.pdf", bbox_inches='tight')
plt.close()

# Achieved QoS per Job
job_ids = [job.id for job in batch.list_jobs]
id_offset = batch.list_jobs[0].id


def get_job_norm_sld(violation):
    norm_sld = []
    for job_id in job_ids:
        job_qos = batch.list_jobs[job_id - id_offset].qos.value
        if job_id in violation.job_sld:
            norm_sld.append(violation.job_sld[job_id] / job_qos)
        else:
            norm_sld.append(1 / job_qos)

    return norm_sld


norm_sld_gpupool = get_job_norm_sld(gpupool_violation)
norm_sld_mig = [job.sld_mig / job.qos.value for job in batch.list_jobs]
norm_sld_heuristic = get_job_norm_sld(heuristic_violation_final)

xs = range(len(batch.list_jobs))

plt.axhline(1, linestyle='dotted')
sns.scatterplot(x=xs, y=norm_sld_mig, label='Coarse', marker='s')
sns.scatterplot(x=xs, y=norm_sld_heuristic, label='Heuristic', marker='<')
sns.scatterplot(x=xs, y=norm_sld_gpupool, label='GPUPool', marker='o')
plt.legend(bbox_to_anchor=(1.25, 1.0))
plt.xlabel("Job ID")
plt.ylabel("Normalized Throughput")

plt.savefig("qos_job.pdf", bbox_inches='tight')
plt.close()
