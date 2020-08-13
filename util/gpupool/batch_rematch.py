from gpupool.workload import BatchJob, GpuPoolConfig
from gpupool.predict import Allocation, StageOne, StageTwo


job_step = 50
min_jobs = 50
max_jobs = 300
max_jobs += job_step
f = open('sensitivity_2.out', 'a')
f.write('num_jobs, (gpupool_required_gpus, gpupool_violations, gpupool_ws),\
mig_required_gpus, random_violations, dram_bw_based_violations\n')
for num_jobs in range(min_jobs, max_jobs, job_step):
    result = []
    # generate required number of jobs
    batch = BatchJob(rand_seed=0, num_jobs=num_jobs)
    pickle_name = "pickles/BatchJob-" + str((num_jobs // 50) - 1) + "-Three_D-BoostTree-Steady-False.pkl"
    print(pickle_name)
    batch.load_df_from_pickle(pickle_name)
    # Get GPUPool results
    config = GpuPoolConfig(Allocation.Three_D, StageOne.BoostTree, StageTwo.Steady,
            at_least_once=False, accuracy_mode=True, stage2_buffer=0.15)

    gpupool, gpupool_viol, gpupool_ws = batch._max_matching(config, cores=32)

    # Get baseline #1: MIG results
    mig = batch.calculate_gpu_count_mig()

    # Get baseline #2: Random matching results
    random = batch.calculate_qos_violation_random(gpupool, 32)

    # Get baseline #3: Dram_bw based results
    dram_bw_based = batch.calculate_qos_viol_dram_bw(gpupool, 32)

    result.append((batch.num_jobs, (gpupool, gpupool_viol.count, gpupool_ws), mig,
        random.count, dram_bw_based))
    f.write('%s\n' % result)

f.close()
