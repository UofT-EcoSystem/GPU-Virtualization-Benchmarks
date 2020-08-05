import argparse
import sys
import multiprocessing as mp
import numpy as np

from gpupool.workload import BatchJob, GpuPoolConfig
from gpupool.predict import Allocation, StageOne, StageTwo


def parse_args():
    parser = argparse.ArgumentParser("Run GPUPool Experiments.")

    parser.add_argument('--exp', default=0, choices=[0, 1, 2],
                        type=int,
                        help='Experiment ID: '
                             '0 = generic test. '
                             '1 = sensitivity analysis for number of unique '
                             'benchmarks per job. '
                             '2 = sensitivity analysis for number of jobs '
                             'per batch. ')
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
    parser.add_argument('--load_model', action='store_true',
                        help='Whether to load boosting tree model from pickle.')
    parser.add_argument('--cores', default=mp.cpu_count(),
                        type=int,
                        help='Number of cores to run on. Default is the '
                             'number of available CPU cores in the system.')

    results = parser.parse_args()

    return results


def main():
    args = parse_args()

    if args.exp == 0:
        # Generic experiment
        num_batches = 5
        num_jobs = 200

        result = []
        for batch_id in range(num_batches):
            batch = BatchJob(rand_seed=batch_id, num_jobs=num_jobs)

            # Get GPUPool results
            gpupool_config = GpuPoolConfig(Allocation.Three_D,
                                           StageOne[args.stage1],
                                           StageTwo[args.stage2],
                                           at_least_once=False,
                                           load_pickle_model=args.load_model)
            gpupool, gpupool_violation = \
                batch.calculate_gpu_count_gpupool(gpupool_config,
                                                  cores=args.cores,
                                                  save=args.save)

            # Get baseline #1: MIG results
            mig = batch.calculate_gpu_count_mig()

            # Get baseline #2: Random matching results
            random_violation = \
                batch.calculate_qos_violation_random(gpupool,
                                                     cores=args.cores)

            result.append((batch.num_jobs, gpupool, mig, random_violation))

            print("=" * 100)
            print("Batch {} with {} jobs:".format(batch_id, batch.num_jobs))
            print("GPUPool: {} GPUs with".format(gpupool),
                  gpupool_violation.to_string())
            print("MIG: {} GPUs".format(mig))
            print("Random: same number of GPUs as GPUPool with",
                  random_violation.to_string())

            print("-" * 100)
            print("Profiling info:")
            for time_name in batch.time_gpupool:
                print("{} {:.6f} sec".format(
                    time_name, batch.time_gpupool[time_name]))
            stage1 = batch.df_pair[gpupool_config.get_perf()].apply(
                lambda x: x.time_stage1)
            stage2 = batch.df_pair[gpupool_config.get_perf()].apply(
                lambda x: x.time_stage2)
            print("Sum of time spent in Stage1: ", sum(stage1) / mp.cpu_count())
            print("Sum of time spent in Stage2: ", sum(stage2) / mp.cpu_count())

            print("MIG time {:.6f} sec".format(batch.time_mig))

        print("=" * 100)
        print("Aggregate results:")
        for num_jobs, gpupool, mig, random_violations in result:
            print("{} jobs used {} GPUs with GPUPool and {} GPUs with MIG "
                  "and {} violations with random."
                  .format(num_jobs, gpupool, mig, random_violations))

        # TODO: how to get an average number?
        result = np.array(result)
        sum_gpupool = sum(result[:, 1])
        sum_mig = sum(result[:, 2])
        pct_reduction = (sum_mig - sum_gpupool) / sum_mig * 100
        print("GPUPool uses {:.2f}% fewer GPUs than MIG on average."
              .format(pct_reduction))

        sum_jobs = sum(result[:, 0])
        sum_violations = sum(result[:, 3])
        print("Random matching introduces {:.2f} violations per job "
<<<<<<< HEAD
              "on average.".format(sum_violations / sum_jobs))
    elif args.exp == 2:
        job_step = 50
        min_jobs = 350
        max_jobs = 350
        max_jobs += job_step
        f = open('sensitivity_2.out', 'a')
        for num_jobs in range(min_jobs, max_jobs, job_step):
            result = []
            # generate required number of jobs
            batch = BatchJob(rand_seed=0, num_jobs=num_jobs)
            # Get GPUPool results
            gpupool_config = GpuPoolConfig(Allocation.Three_D,
                                           StageOne[args.stage1],
                                           StageTwo[args.stage2],
                                           at_least_once=False,
                                           load_pickle_model=args.load_model)
            gpupool, gpupool_viol = batch.calculate_gpu_count_gpupool(
                gpupool_config, save=args.save)

            # Get baseline #1: MIG results
            mig = batch.calculate_gpu_count_mig()

            # Get baseline #2: Random matching results
            random = batch.calculate_qos_violation_random(gpupool)

            result.append((batch.num_jobs, (gpupool, gpupool_viol), mig, random))
            f.write('%s\n' % result)

        f.close()


=======
              "on average.".format(sum_violations.count / sum_jobs))
>>>>>>> beaa4b2261c648f2cceadd241703b2424a8f2557
    else:
        print("Unimplemented Error.")
        sys.exit(1)

    # batch = BatchJob(rand_seed=0)
    # batch.df_pair[batch.df_pair['pair_str'] == 'job-0+job-364'].iloc[0][
    #     'pair_job'].get_performance(Allocation.Three_D, StageOne.GPUSim,
    #                                 StageTwo.Full, 4, False)


if __name__ == '__main__':
    main()
