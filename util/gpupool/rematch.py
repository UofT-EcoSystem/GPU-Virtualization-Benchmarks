from gpupool.workload import BatchJob, GpuPoolConfig
from gpupool.predict import Allocation, StageOne, StageTwo

batch = BatchJob(rand_seed=0, num_jobs=50)
batch.load_df_from_pickle(
    "pickles/BatchJob-0-Three_D-BoostTree-Steady-False.pkl")

pair = batch.df_pair[batch.df_pair['pair_str'] == 'job-19+job-25'][
    'pair_job'].iloc[0]

config = GpuPoolConfig(Allocation.Three_D, StageOne.BoostTree, StageTwo.Steady,
                       at_least_once=False, accuracy_mode=False,
                       stage2_buffer=0.05)
result = pair.get_gpupool_performance(config)
print("ws", result[config.get_ws()])
print("sld", result[config.get_perf()].sld)
result[config.get_option()].print_interference()
option = result[config.get_option()]

# count, violation, ws = batch._max_matching(config, cores=32)
#
#
# print("Count", count)
# print(violation.to_string(20))
# print("WS", ws)
from copy import deepcopy
good_option = deepcopy(option)

perf = good_option.get_real_performance()
good_option.print_interference()
print("fixed: ", perf.sld)
print("fixed ws:", sum(perf.sld))

delta = [old - good for old, good in zip(option.interference_matrix,
                                         good_option.interference_matrix)]
option._pretty_print_matrix("delta ", delta)

# good_config = GpuPoolConfig(Allocation.Three_D, StageOne.GPUSim,
#                             StageTwo.Steady,
#                             at_least_once=False, accuracy_mode=False,
#                             stage2_buffer=0.05)
# result = pair.get_gpupool_performance(good_config)
# print("good ws", result[good_config.get_ws()])
# print("good sld", result[good_config.get_perf()].sld)
# result[good_config.get_option()].print_interference()

