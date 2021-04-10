from gpupool.core.workload import BatchJob, GpuPoolConfig
from gpupool.core.predict import Allocation, StageOne, StageTwo

num_jobs = 100
batch = BatchJob(rand_seed=0, num_jobs=num_jobs)
batch.load_df_from_pickle(
    "pickles/BatchJob-0-Three_D-BoostTree-Steady-False.pkl")


config = GpuPoolConfig(Allocation.Three_D, StageOne.BoostTree, StageTwo.Steady,
                       at_least_once=False, accuracy_mode=False,
                       stage2_buffer=0.1)

# count, violation, ws = batch._max_matching(config, cores=32)
#
# print("GPU count:", count)
# print("WS:", ws)
# print(violation.to_string(num_jobs))

pair = batch.df_pair[batch.df_pair['pair_str'] == 'job-12+job-28'][
    'pair_job'].iloc[0]
result = pair.get_gpupool_performance(config)
print("ws", result[config.get_ws()])
print("sld", result[config.get_perf()].sld)
result[config.get_option()].print_interference()
option = result[config.get_option()]

from copy import deepcopy
good_option = deepcopy(option)

perf = good_option.get_real_performance()
good_option.print_interference()
print("fixed: ", perf.sld)
print("fixed ws:", sum(perf.sld))

delta = [old - good for old, good in zip(option.interference_matrix,
                                         good_option.interference_matrix)]
option._pretty_print_matrix("delta ", delta)

