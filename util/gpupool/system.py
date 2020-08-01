import argparse
import sys

from gpupool.workload import BatchJob
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

    results = parser.parse_args()

    return results


def main():
    args = parse_args()

    if args.exp == 0:
        # Generic experiment
        num_batches = 1
        for batch_id in range(num_batches):
            batch = BatchJob(rand_seed=batch_id, num_jobs=100)
            gpupool = batch.calculate_gpu_count_gpupool(Allocation.Three_D,
                                                        StageOne[args.stage1],
                                                        StageTwo[args.stage2],
                                                        at_least_once=False,
                                                        save=args.save)
            # mig = batch.calculate_gpu_count_mig()

            print("Batch {}:".format(batch_id))
            print("GPUPool: {} GPUs", gpupool)
            # print("MIG: {} GPUs", mig)

    else:
        print("Unimplemented Error.")
        sys.exit(1)

    # batch = BatchJob(rand_seed=0)
    # batch.df_pair[batch.df_pair['pair_str'] == 'job-0+job-364'].iloc[0][
    #     'pair_job'].get_performance(Allocation.Three_D, StageOne.GPUSim,
    #                                 StageTwo.Full, 4, False)


if __name__ == '__main__':
    main()
