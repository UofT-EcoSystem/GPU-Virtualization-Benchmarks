import argparse
import sys
import multiprocessing as mp
import numpy as np

from gpupool.workload import BatchJob, GpuPoolConfig
from gpupool.predict import Allocation, StageOne, StageTwo
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Run GPUPool Experiments.")

    parser.add_argument('--exp', default=0, choices=[0, 1, 2, 3],
                        type=int,
                        help='Experiment ID: '
                             '0 = generic test. '
                             '1 = sensitivity analysis for number of unique '
                             'benchmarks per job. '
                             '2 = sensitivity analysis for number of jobs '
                             'per batch. '
                             '3 = boosting tree test.')
    parser.add_argument('--stage1', default="GPUSim",
                        choices=[s1.name for s1 in StageOne],
                        help='Stage 1 predictor: GPUSim or BoostTree.')
    parser.add_argument('--stage2', default="Full",
                        choices=[s2.name for s2 in StageTwo],
                        help='Stage 2 predictor: Full or Steady or Weighted '
                             'or GPUSim.')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save df_pair into pickles for each '
                             'simulated batch job.')
    parser.add_argument('--cores', default=mp.cpu_count(),
                        type=int,
                        help='Number of cores to run on. Default is the '
                             'number of available CPU cores in the system.')
    parser.add_argument('--accuracy_mode', action='store_true',
                        help='Only relevant when stage1 is BoostTree. '
                             'Accuracy mode will do cross validation and '
                             'train a new model for each pair of jobs.'
                             'Otherwise, runtime mode will simply load a '
                             'trained model to do fast inference.')
    parser.add_argument('--stage2_buffer', default=0.05,
                        type=float,
                        help='Amount of buffer to tighten qos check in stage2.')

    results = parser.parse_args()

    return results


def run_exp_0(args):
    # Generic experiment
    num_batches = 5
    num_jobs = 100

    result = []
    for batch_id in range(num_batches):
        batch = BatchJob(rand_seed=batch_id, num_jobs=num_jobs)

        # Get GPUPool results
        gpupool_config = GpuPoolConfig(Allocation.Three_D,
                                       StageOne[args.stage1],
                                       StageTwo[args.stage2],
                                       at_least_once=False,
                                       accuracy_mode=args.accuracy_mode,
                                       stage2_buffer=args.stage2_buffer)
        gpupool, gpupool_violation, gpupool_ws_total = \
            batch.calculate_gpu_count_gpupool(gpupool_config,
                                              cores=args.cores,
                                              save=args.save)

        # Get baseline #1: MIG results
        mig, mig_ws_total, ws_list = batch.calculate_gpu_count_mig()

        # Get baseline #2: Random matching results
        random_violation = \
            batch.calculate_qos_violation_random(gpupool,
                                                 cores=args.cores)

        result.append(
            {"batch": batch,
             "gpupool": gpupool,
             "gpupool_violation": gpupool_violation,
             "gpupool_weighted_speedup": gpupool_ws_total,
             "mig": mig,
             "random_violation": random_violation
             }
        )

        print("=" * 100)
        print("Batch {} with {} jobs:".format(batch_id, batch.num_jobs))
        print("GPUPool: {} GPUs achieving {} weighted speedup with".
                format(gpupool, gpupool_ws_total),
              gpupool_violation.to_string(batch.num_jobs))
        print("MIG: {} GPUs achieving {} weighted speedup".format(mig,
            mig_ws_total))
        print("Random: same number of GPUs as GPUPool with",
              random_violation.to_string(batch.num_jobs))

        print("-" * 100)
        print("Profiling info:")
        for time_name in batch.time_gpupool:
            print("{} {:.6f} sec".format(
                time_name, batch.time_gpupool[time_name]))
        stage1 = batch.df_pair[gpupool_config.get_perf()].apply(
            lambda x: x.time_stage1)
        stage2 = batch.df_pair[gpupool_config.get_perf()].apply(
            lambda x: x.time_stage2)
        stage2_steady = batch.df_pair[gpupool_config.get_perf()].apply(
            lambda x: x.steady_iter[0])
        stage2_steady_2 = batch.df_pair[gpupool_config.get_perf()].apply(
            lambda x: x.steady_iter[1]
        )
        stage_steady = pd.concat([stage2_steady, stage2_steady_2])

        print("Sum of time spent in Stage1: ", sum(stage1) / mp.cpu_count())
        print("Sum of time spent in Stage2: ", sum(stage2) / mp.cpu_count())

        print("Stage2 std: ", np.std(stage2))
        print("Stage2 mean: ", np.mean(stage2))
        print("Stage2 steady std: ", np.std(stage2_steady))
        print("Stage2 steady mean: ", np.mean(stage_steady))

        print("MIG time {:.6f} sec".format(batch.time_mig))

    # Aggregate results over all runs
    print("=" * 100)
    print("Aggregate results:")

    # TODO: how to get an average number?

    sum_gpupool = sum([r['gpupool'] for r in result])
    sum_mig = sum(r['mig'] for r in result)

    # GPUPool v.s. baseline #1
    pct_reduction = (sum_mig - sum_gpupool) / sum_mig * 100
    print("GPUPool uses {:.2f}% fewer GPUs than MIG on average."
          .format(pct_reduction))

    # GPUPool v.s. baseline #2
    sum_jobs = sum([r['batch'].num_jobs for r in result])
    sum_gpupool_violations = sum([r['gpupool_violation'] for r in result])
    sum_random_violations = sum([r['random_violation'] for r in result])

    print("GPUPool:", sum_gpupool_violations.to_string(sum_jobs))
    print("Random matching:", sum_random_violations.to_string(sum_jobs))

    if sum_gpupool_violations.count:
        print("Random matching introduces {:.2f}% more violations than "
              "GPUPool.".format((sum_random_violations.count -
                                 sum_gpupool_violations.count) /
                                sum_gpupool_violations.count * 100))


def run_exp_2(args):

    job_step = 50
    min_jobs = 50
    max_jobs = 350
    max_jobs += job_step
    f = open('window_sensitivity.txt', 'w')
    f.write('num_jobs, gpupool_required_gpus, gpupool_violations,'
            'mig_required_gpus, heuristic_gpus\n')
    for num_jobs in range(min_jobs, max_jobs, job_step):
        seed = num_jobs // job_step
        print(seed)
        # generate required number of jobs
        batch = BatchJob(rand_seed=seed, num_jobs=num_jobs)
        # Get GPUPool results
        gpupool_config = GpuPoolConfig(Allocation.Three_D,
                                       StageOne[args.stage1],
                                       StageTwo[args.stage2],
                                       at_least_once=False,
                                       accuracy_mode=args.accuracy_mode,
                                       stage2_buffer=args.stage2_buffer)
        gpupool, gpupool_viol, gpupool_ws = batch.calculate_gpu_count_gpupool(
            gpupool_config, args.cores, save=args.save)

        # Get baseline #1: MIG results
        mig, mig_ws, mig_ws_list = batch.calculate_gpu_count_mig()

        # Get baseline #2: Dram_bw based results
        violation_dram_bw, dram_bw_ws_list, migrated_count_dram = \
            batch.calculate_qos_viol_dram_bw(gpupool, args.cores)

        f.write('{},{},{},{},{}\n'.format(batch.num_jobs,
                                          gpupool,
                                          gpupool_viol.count,
                                          mig,
                                          migrated_count_dram))

    f.close()


def run_exp_3(args):
    # Boosting tree test
    random_seed = 1000
    num_jobs = 200

    batch = BatchJob(rand_seed=random_seed, num_jobs=num_jobs)

    gpupool_config = GpuPoolConfig(Allocation.Three_D,
                                   StageOne[args.stage1],
                                   StageTwo[args.stage2],
                                   at_least_once=False,
                                   accuracy_mode=args.accuracy_mode,
                                   stage2_buffer=args.stage2_buffer)

    error = batch.boosting_tree_test(gpupool_config, args.cores)

    print("Boosting tree test average relative error: {:.2f}%"
          .format(error * 100))


def main():
    args = parse_args()

    if args.exp == 0:
        run_exp_0(args)
    elif args.exp == 2:
        run_exp_2(args)
    elif args.exp == 3:
        run_exp_3(args)
    else:
        print("Unimplemented Error.")
        sys.exit(1)

    # batch = BatchJob(rand_seed=0)
    # batch.df_pair[batch.df_pair['pair_str'] == 'job-0+job-364'].iloc[0][
    #     'pair_job'].get_performance(Allocation.Three_D, StageOne.GPUSim,
    #                                 StageTwo.Full, 4, False)


if __name__ == '__main__':
    main()
