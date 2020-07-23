from gpupool.workload import BatchJob
from gpupool.predict import Allocation, StageOne, StageTwo


def main():
    batch = BatchJob(200, num_benchmarks_per_job=10, num_jobs=4)
    batch.calculate_gpupool_performance(Allocation.One_D,
                                        StageOne.GPUSim,
                                        StageTwo.Full,
                                        at_least_once=True)
    print(batch.df_pair)


if __name__ == '__main__':
    main()
