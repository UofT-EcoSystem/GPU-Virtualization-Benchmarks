import argparse
import sys
import multiprocessing as mp
import numpy as np
from scipy import stats

from gpupool.core.workload import BatchJob, GpuPoolConfig
from gpupool.core.predict import Allocation, StageOne, StageTwo
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
    parser.add_argument('--job_count', default=100, type=int, 
                        help='job count for exp0.')
    parser.add_argument('--batch', default=5, type=int,
                        help="Number of batches for exp 0")
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
    parser.add_argument('--profile_stage1', action='store_true',
                        help='whether to only run stage 1 for profiling')
    parser.add_argument('--stage2_buffer', default=0.05,
                        type=float,
                        help='Amount of buffer to tighten qos check in stage2.')
    parser.add_argument('--system', nargs='+',
                        default=['gpupool', 'mig', 'heuristic'],
                        help="Which systems to evaluate.")
    parser.add_argument('--seed', type=int,
                        default=0, help="starting seed for generating jobs")

    results = parser.parse_args()

    return results


def run_exp_0(args):
    # Generic experiment
    num_batches = args.batch
    num_jobs = args.job_count

    results = []
    for batch_id in range(num_batches):
        from gpupool.core.workload import Job
        Job.count = 0
        batch = BatchJob(rand_seed=batch_id + args.seed,
                         num_jobs=num_jobs)
        batch_result = {'batch': batch}

        if 'gpupool' in args.system:
            # Get GPUPool results
            gpupool_config = GpuPoolConfig(Allocation.Three_D,
                                           StageOne[args.stage1],
                                           StageTwo[args.stage2],
                                           at_least_once=False,
                                           accuracy_mode=args.accuracy_mode,
                                           stage2_buffer=args.stage2_buffer,
                                           profile_stage1=args.profile_stage1)
            gpupool = batch.calculate_gpu_count_gpupool(gpupool_config,
                                                        cores=args.cores,
                                                        save=args.save)

            if args.profile_stage1:
                continue

            print("-" * 100)
            print("Profiling info:")
            print("Two-stage predictor took", gpupool['pred_latency'])

            stage1 = batch.df_pair[gpupool_config.get_perf()].apply(
                lambda x: x.time_stage1)
            stage2 = batch.df_pair[gpupool_config.get_perf()].apply(
                lambda x: x.time_stage2)

            print("Sum of time spent in Stage1: ", sum(stage1) / mp.cpu_count())
            print("Sum of time spent in Stage2: ", sum(stage2) / mp.cpu_count())

            print("Stage2 latency mean: ", np.mean(stage2))
            print("Stage2 latency std: ", np.std(stage2))

            if args.stage2 == StageTwo.Steady:
                stage2_iters = batch.df_pair[gpupool_config.get_perf()].apply(
                    lambda x: x.steady_iter[0])
                stage2_iters_2 = batch.df_pair[gpupool_config.get_perf()].apply(
                    lambda x: x.steady_iter[1]
                )
                stage_steady = pd.concat([stage2_iters, stage2_iters_2])
                print("Stage2 steady std: ", np.std(stage_steady))
                print("Stage2 steady mean: ", np.mean(stage_steady))

            batch_result["gpupool_count"] = gpupool['gpu_count']
            batch_result["gpupool_violation"] = gpupool['violation']
            batch_result["gpupool_ws"] = gpupool['ws_total']
            batch_result["gpupool_margin"] = gpupool['margin']

        if 'mig' in args.system:
            # Get baseline #1: MIG results
            mig = batch.calculate_gpu_count_mig()

            batch_result["mig_count"] = mig['gpu_count']
            batch_result["mig_ws"] = mig['ws_total']
            # print("MIG time {:.6f} sec".format(batch.time_mig))

        if 'heuristic' in args.system:
            # Get baseline #2: memory bandwidth results
            heuristic = batch.calculate_qos_viol_dram_bw(gpu_avail=50,
                                                  cores=args.cores)

            batch_result["heuristic_count"] = heuristic['gpu_count']
            batch_result["heuristic_ws"] = heuristic['ws_total']

        print(batch_result)
        results.append(batch_result)

    # Aggregate results over all runs
    print("=" * 100)
    print(results)

    if num_batches > 1:
        print("Aggregate results:")

        for system in args.system:
            print('*' * 10, system, '*' * 10)

            counts = [r['{}_count'.format(system)] for r in results]
            print("count avg = {}, ste = {}".format(np.average(counts),
                                                    stats.sem(counts)))

            stp = [r['{}_ws'.format(system)] for r in results]
            print("stp avg = {}, ste = {}".format(np.average(stp),
                                                  stats.sem(stp)))


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
