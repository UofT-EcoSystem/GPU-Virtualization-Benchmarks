from gpupool.core.workload import GpuPoolConfig, Violation
import time
from copy import deepcopy
import pandas as pd
import numpy as np
import multiprocessing as mp


def two_stage_predict(df, config: GpuPoolConfig):
    # Deep copy config so that each process can have a unique model copy to
    # run on if stage1 is boosting tree
    option_copy = deepcopy(config)

    start = time.perf_counter()

    _df_performance = df['pair_job'].apply(
        lambda pair_job:
        pd.Series(pair_job.get_gpupool_performance(option_copy))
    )

    duration = time.perf_counter() - start
    print("each core sec", duration)

    return _df_performance


def get_qos_violations(job_pairs: list):
    violation = Violation()

    for list_job in job_pairs:
        qos_goals = [list_job[0].qos.value, list_job[1].qos.value]

        from gpupool.core.predict import PairJob
        pair = PairJob([list_job[0], list_job[1]])
        perf = pair.get_best_effort_performance()
        violated_pair = False

        for sld, qos in zip(perf.sld, qos_goals):
            violated_pair |= violation.update(sld, qos)

        if violated_pair:
            violation.gpu_increase += 1
            violation.actual_ws += 2
            violation.actual_ws_list += [1, 1]

            # Update job slowdown for heuristic
            for job in list_job:
                violation.job_sld[job.id] = 1
        else:
            violation.actual_ws += sum(perf.sld)
            violation.actual_ws_list.append(sum(perf.sld))

            # Update job slowdown for heuristic
            for job, sld in zip(list_job, perf.sld):
                violation.job_sld[job.id] = sld
    return violation


def verify_qos(options: list):
    violation = Violation()

    for option in options:
        job_pair = option.jobs
        qos_goals = [job_pair[0].qos.value, job_pair[1].qos.value]

        # This function changes interference matrix get a deepcopy first
        from copy import deepcopy
        option_copy = deepcopy(option)
        perf = option_copy.get_real_performance()

        violated_pair = False
        for sld, qos in zip(perf.sld, qos_goals):
            violated_pair |= violation.update(sld, qos)

        if violated_pair:
            violation.gpu_increase += 1
            violation.actual_ws += 2
            violation.actual_ws_list += [1, 1]

            # Update job slowdown for GPUPool
            for job in option.jobs:
                violation.job_sld[job.id] = 1
        else:
            violation.actual_ws += sum(perf.sld)
            violation.actual_ws_list.append(sum(perf.sld))

            # Update job slowdown for GPUPool
            for job, sld in zip(option.jobs, perf.sld):
                violation.job_sld[job.id] = sld

        violation.ws_no_migrate += sum(perf.sld)

    return violation


def verify_boosting_tree(df, config: GpuPoolConfig):
    config_copy = deepcopy(config)

    delta = df['pair_job'].apply(
        lambda pair_job: pair_job.verify_boosting_tree(config_copy)
    )

    return delta.tolist()


def parallelize(data, cores, func, extra_param=None):
    cores = min(cores, len(data))
    pool = mp.Pool(cores)

    data_split = np.array_split(data, cores)

    if extra_param:
        data_split = [(ds, extra_param) for ds in data_split]
        result = pool.starmap(func, data_split)
    else:
        result = pool.map(func, data_split)

    pool.close()
    pool.join()

    return result
