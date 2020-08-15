import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from gpupool.workload import BatchJob, GpuPoolConfig
from gpupool.predict import Allocation, StageOne, StageTwo

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
qos_violations = num_jobs
heuristic_count = num_jobs / 2

heuristic_migrated_count_final = num_jobs
ws_list_heuristic_final = []
heuristic_count_final = num_jobs
# Iteratively find the best config
while qos_violations > 0:
    heuristic_count += 2
    qos_violations, ws_list, gpu_migrated = \
        batch.calculate_qos_viol_dram_bw(heuristic_count,
                                         cores=32)
    if gpu_migrated < heuristic_migrated_count_final:
        heuristic_migrated_count_final = gpu_migrated
        ws_list_heuristic_final = ws_list
        heuristic_count_final = heuristic_count

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
plt.xlabel("GPU ID")
plt.ylabel("Weighted Speedup")
plt.ylim([1, 2])

plt.savefig("ws_gpu.pdf", bbox_inches='tight')
plt.close()

# Draw bar graphs for comparison
plt.figure(figsize=(4, 5))
ax = sns.barplot(x=["No-Sharing", "Coarse-Grained", "Heuristic", "GPUPool"],
            y=[num_jobs, mig_count, heuristic_migrated_count_final,
               gpupool_count],
            )
plt.ylabel("Required GPU Count")
plt.xticks(rotation=30)
plt.savefig("required_gpu_count.pdf", bbox_inches='tight')
hatches = itertools.cycle(['//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
for i, bar in enumerate(ax.patches):
    hatch = next(hatches)
    bar.set_hatch(hatch)
plt.close()

# Draw aggregate WS stats
plt.figure(figsize=(4, 5))
ax = sns.barplot(x=["No-Sharing", "Coarse-Grained", "Heuristic", "GPUPool"],
                 y=[1, mig_ws, heuristic_ws_final, gpupool_ws]
                 )
plt.ylabel("STP")
plt.xticks(rotation=30)
plt.savefig("aggregate_ws.pdf", bbox_inches='tight')
hatches = itertools.cycle(['//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
for i, bar in enumerate(ax.patches):
    hatch = next(hatches)
    bar.set_hatch(hatch)
plt.close()

