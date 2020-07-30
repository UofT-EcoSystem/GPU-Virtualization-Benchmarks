from random import seed
from random import choices
from enum import Enum
import pandas as pd
import sys
import pickle
import os
import subprocess
import numpy as np

import data.scripts.common.constants as const
from gpupool.predict import Allocation, StageOne, StageTwo

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


class QOS(Enum):
    PCT_50 = 0.5
    PCT_60 = 0.6
    PCT_70 = 0.7
    PCT_80 = 0.8
    PCT_90 = 0.9


class ITER(Enum):
    I_50 = 50
    I_100 = 100
    I_200 = 200
    I_400 = 400
    I_800 = 800
    I_1600 = 1600


class Job:
    count = 0
    job_list = {}

    def __init__(self, qos: QOS, num_iters, benchmarks: list):
        self.qos = qos
        self.num_iters = num_iters

        # Create new job with a list of benchmarks
        self.benchmarks = benchmarks

        self.id = Job.count
        Job.count += 1

        self.name = "job-{}".format(self.id)

        Job.job_list[self.name] = self.benchmarks

    def get_seq_cycles(self):
        return const.get_seq_cycles(self.name)

    def get_num_benchmarks(self):
        return len(self.benchmarks)

    def calculate_static_partition(self):
        mig_pickles = os.path.join(THIS_DIR, '../data/pickles/mig_ipcs')
        mig_ipcs = pickle.load( open(mig_pickles, 'rb'))
        cycle_lengths = self.get_seq_cycles()

        full_runtime = sum(cycle_lengths)
        for resources in range(8):
            # calculate qos with i static partitions available
            restricted_runtime = 0
            for bm_index in range(len(self.benchmarks)):
                # calculate how much time will be added by restricting this 
                # benchmark
                restricted_runtime += cycle_lengths[bm_index] / \
                        mig_ipcs[self.benchmarks[bm_index]][resources]
            # use restricted runtime to estimate qos at this static partition
            attained_qos = full_runtime / restricted_runtime
            if (attained_qos >= self.qos.value):
                break
        return resources + 1


class BatchJob:
    REPEAT = 0
    FRESH = 1

    ITER_CHOICES = [50, 100, 200, 400, 800, 1600]

    count = 0

    def __init__(self, rand_seed, num_benchmarks_per_job, num_jobs,
                 allow_repeat=False, p_repeat=0.2):
        self.rand_seed = rand_seed
        self.num_benchmarks_per_job = num_benchmarks_per_job
        self.num_jobs = num_jobs
        self.allow_repeat = allow_repeat
        self.p_repeat = p_repeat

        self.id = BatchJob.count
        BatchJob.count += 1

        self.list_jobs = []
        self._synthesize_jobs()

        self.df_pair = pd.DataFrame()
        self._create_pairs()

    def calculate_gpupool_performance(self, alloc: Allocation,
                                      stage_1: StageOne, stage_2: StageTwo,
                                      num_slices=4,
                                      at_least_once=False):
        df_performance = self.df_pair['pair_job'].apply(
            lambda pair_job: pd.Series(pair_job.get_performance(alloc,
                                                                stage_1,
                                                                stage_2,
                                                                num_slices,
                                                                at_least_once))
        )

        # Drop the same columns
        drop_col = [col for col in self.df_pair.columns
                    if col in df_performance.columns]
        self.df_pair.drop(columns=drop_col, inplace=True)

        self.df_pair = self.df_pair.merge(df_performance,
                                          left_index=True, right_index=True)

    def _synthesize_jobs(self):
        self.list_jobs.clear()
        seed(self.rand_seed)

        # Candidates are non multi-kernel benchmarks
        benchmarks = [benchmark for benchmark in const.kernel_yaml if benchmark
                      not in const.multi_kernel_app]

        for app_idx in range(self.num_jobs):
            list_benchmarks = []

            # Generate QoS for the job
            qos = choices(list(QOS))[0]

            # Generate number of iterations for the job
            num_iters = choices(BatchJob.ITER_CHOICES)[0]

            # Generate benchmarks within the job
            for bench_idx in range(self.num_benchmarks_per_job):
                if bench_idx == 0:
                    # The first one cannot be repeat
                    list_benchmarks += choices(benchmarks)
                else:
                    # Figure out if we are repeating or selecting
                    # a fresh benchmark
                    repeat_or_fresh = choices([BatchJob.REPEAT, BatchJob.FRESH],
                                              weights=[self.p_repeat,
                                                       1 - self.p_repeat])[0]
                    if self.allow_repeat and repeat_or_fresh == BatchJob.REPEAT:
                        list_benchmarks.append('repeat')
                    else:
                        # Make sure we don't pick the same benchmark again
                        leftover = [benchmark for benchmark in benchmarks
                                    if benchmark not in list_benchmarks]
                        list_benchmarks += choices(leftover)

            new_job = Job(qos, num_iters, benchmarks=list_benchmarks)
            self.list_jobs.append(new_job)

    def _create_pairs(self):
        from gpupool.predict import PairJob
        pairs = [PairJob([lhs, rhs])
                 for lhs in self.list_jobs for rhs in self.list_jobs
                 if lhs.name < rhs.name]

        self.df_pair['pair_job'] = pairs
        self.df_pair['pair_str'] = self.df_pair['pair_job'].apply(lambda x:
                                                                  x.name())

    def calculate_gpu_count_mig(self):
        # convert jobs to size slices 
        jobs = [x.calculate_static_partition() for x in self.list_jobs]
        indeces = np.argsort(jobs)[::-1]
        jobs = np.sort(jobs)[::-1]
        if np.where(jobs <= 3)[0].size != 0:
            print("There are jobs that require less than 4 slices of GPU")
        num_gpus = 0
        f = open("mig_sched.out", "a")


        # returns the index of the element that is to fill the remaining slice
        def best_fill(jobs: list, rem_slice: int) -> int:
            # find elements that can fill the spot
            suitable_indeces = np.where(jobs <= rem_slice) 
            suitable_values = np.take(jobs, suitable_indeces)
            if suitable_values[0].size != 0:
                return np.argmax(suitable_values) 
            return -1

        # fill the machine as much as possible with suitable slices
        def fill_machine(jobs: list, indeces: list, rem_slice: int):
            #print("inside fill machine, rem_slice is ", rem_slice)
            # check if there are suitable jobs in the array and fill
            while np.where(jobs <= rem_slice)[0].size != 0:
                #print("suitable jobs are ", np.where(jobs <= rem_slice)[0])
                fill_index = best_fill(jobs, rem_slice)
                #print("fill_index is ", fill_index)
                if fill_index == -1:
                    break
                
                # remove element at fill_index and update jobs and indeces
                job_number = indeces[fill_index]
                job_slice = jobs[fill_index]
                job_line = str(job_number) + " "
                f.write(job_line)
                indeces = np.delete(indeces, fill_index)
                jobs = np.delete(jobs, fill_index)
                rem_slice -= job_slice
                #print("remaining slice on this gpu is ", rem_slice)
            return jobs, indeces


        job_sizes = "Job sizes   " + np.array_str(jobs) + "\n"
        job_indeces = "Job indeces " + np.array_str(indeces) + "\n" + "\n"
        f.write(job_sizes)
        f.write(job_indeces)

        # main algo loop where we fill machines with jobs
        while len(jobs) > 0:

            #fill next machine
            num_gpus += 1
            gpu_line = "GPU " + str(num_gpus) + ": Jobs "
            f.write(gpu_line)
            jobs, indeces = fill_machine(jobs, indeces, 8)
            #print("after gpu is filled we have ", jobs, indeces, num_gpus)
            f.write("\n")

        
        f.close()
        return num_gpus


    def calculate_gpu_count_gpupool(self, alloc, stage_1: StageOne, 
                        stage_2: StageTwo,
                        at_least_once):
        combo_name = "{}-{}-{}-{}".format(alloc.name,
                                          stage_1.name,
                                          stage_2.name,
                                          at_least_once)
        option_col = combo_name + "-" + "option"
        perf_col = combo_name + "-" + "perf"
        ws_col = combo_name + "-" + "ws"
        f = open("input.js", "a")
        f.write("var blossom = require('./edmonds-blossom');" + "\n")
        f.write("var data = [" + "\n")

        # iterate over rows and read off the weighted speedups
        for index, row in self.df_pair.iterrows():
            # determine if we include this pair as an edge or not based on if the 
            # qos of each job in pair is satisfied
            if ((row['pair_job'].jobs[0].qos.value > row[perf_col].sld[0]) or
                    (row['pair_job'].jobs[1].qos.value > row[perf_col].sld[1])):
                continue
            string_pair = row['pair_str']
            job_indeces = string_pair.split('+')
            first_job_index = string_pair.split('+')[0].split('-')[1]
            second_job_index = string_pair.split('+')[1].split('-')[1]
            pair_weighted_speedup = row[ws_col]
            input_line = "      [" + str(first_job_index) + ', ' +  \
                    str(second_job_index) + ', ' \
                    + str(round(pair_weighted_speedup, 3)) + '],\n'
            f.write(input_line)
                
        f.write("    ];\n")
        f.write("var results = blossom(data);\n")
        f.write("console.log(results);\n")
        f.close()
        print('Resulting input file is:')
        print('')
        os.system('cat input.js')
        print('')
        matching = subprocess.getoutput("node input.js")
        matching = matching.replace(" ", "").replace("]", "").replace("[", "")
        matching = matching.split(",")
        matching = [int(x) for x in matching]
        num_pairs = len([x for x in matching if x >= 0]) // 2
        num_isolated = len([x for x in matching if x == -1])
        os.system('rm input.js')
        return num_pairs + num_isolated






































