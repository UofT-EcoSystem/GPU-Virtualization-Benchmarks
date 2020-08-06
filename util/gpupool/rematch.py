from gpupool.workload import BatchJob, GpuPoolConfig
from gpupool.predict import Allocation, StageOne, StageTwo

batch = BatchJob(rand_seed=0, num_jobs=100)
batch.load_df_from_pickle("pickles/BatchJob-0-Three_D-BoostTree-Steady-False.pkl")

config = GpuPoolConfig(Allocation.Three_D, StageOne.BoostTree, StageTwo.Steady, at_least_once=False, accuracy_mode=True, stage2_buffer=0.2)
count, violation = batch._max_matching(config, cores=64)

print("Count", count)
print(violation.to_string(100))

