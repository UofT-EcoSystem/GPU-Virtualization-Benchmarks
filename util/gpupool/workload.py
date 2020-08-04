from random import seed
from random import choices
from enum import Enum
import pandas as pd
import pickle
import os
import subprocess
import numpy as np
import multiprocessing as mp
import time
import sys

import data.scripts.common.constants as const
from gpupool.predict import Allocation, StageOne, StageTwo

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

MAX_PARTITIONS = 8


class QOS(Enum):
    PCT_50 = 0.5
    PCT_60 = 0.6
    PCT_70 = 0.7
    PCT_80 = 0.8
    PCT_90 = 0.9


class GpuPoolConfig:
    def __init__(self, alloc: Allocation, stage_1: StageOne, stage_2: StageTwo,
                 at_least_once):
        self.alloc = alloc
        self.stage_1 = stage_1
        self.stage_2 = stage_2
        self.at_least_once = at_least_once

    def to_string(self):
        combo_name = "{}-{}-{}-{}".format(self.alloc.name,
                                          self.stage_1.name,
                                          self.stage_2.name,
                                          self.at_least_once)
        return combo_name

    def get_ws(self):
        return self.to_string() + '-ws'

    def get_perf(self):
        return self.to_string() + '-perf'

    def get_option(self):
        return self.to_string() + '-option'

    def get_time_prediction(self):
        return self.to_string() + '-time-prediction'

    def get_time_matching(self):
        return self.to_string() + '-time-matching'


# Top-level functions to be pickled and parallelized on multiple cores
def get_perf(df, option: GpuPoolConfig):
    _df_performance = df['pair_job'].apply(
        lambda pair_job:
        pd.Series(pair_job.get_gpupool_performance(option))
    )
    return _df_performance


def get_qos_violations(job_pairs: list):
    violation_count = 0
    for pair in job_pairs:
        qos_goals = [pair[0].qos.value, pair[1].qos.value]

        from gpupool.predict import PairJob
        pair = PairJob([pair[0], pair[1]])
        perf = pair.get_best_effort_performance()

        for sld, qos in zip(perf.sld, qos_goals):
            if sld < qos:
                violation_count += 1

    return violation_count
# End parallel functions


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
        mig_ipcs = pickle.load(open(mig_pickles, 'rb'))
        cycle_lengths = self.get_seq_cycles()

        full_runtime = sum(cycle_lengths)
        for resources in range(MAX_PARTITIONS):
            # calculate qos with i static partitions available
            restricted_runtime = 0
            for bm_index in range(len(self.benchmarks)):
                # calculate how much time will be added by restricting this
                # benchmark
                restricted_runtime += cycle_lengths[bm_index] / \
                        mig_ipcs[self.benchmarks[bm_index]][resources]
            # use restricted runtime to estimate qos at this static partition
            attained_qos = full_runtime / restricted_runtime
            if attained_qos >= self.qos.value:
                return resources + 1

        return MAX_PARTITIONS


class BatchJob:
    REPEAT = 0
    FRESH = 1

    ITER_CHOICES = [400, 800, 1200, 1600, 2000, 2400, 2800]
    NUM_JOB_CHOICES = [100, 200, 300, 400]
    NUM_BENCH_CHOICES = [4, 8, 12, 16]

    count = 0

    def __init__(self, rand_seed, num_benchmarks_per_job=-1, num_jobs=-1,
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

        # In seconds
        self.time_gpupool = {}
        self.time_mig = 0

    def _calculate_gpupool_performance(self, config: GpuPoolConfig):
        def parallelize(df, func):
            cores = mp.cpu_count()
            data_split = np.array_split(df, cores)
            pool = mp.Pool(cores)
            data = pd.concat(
                pool.starmap(func, [(ds, config) for ds in data_split])
            )
            pool.close()
            pool.join()

            return data

        df_performance = parallelize(self.df_pair, get_perf)

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

        if self.num_jobs == -1:
            self.num_jobs = choices(BatchJob.NUM_JOB_CHOICES)[0]

        print("BatchJob {} synthesizes {} jobs.".format(self.id, self.num_jobs))

        for app_idx in range(self.num_jobs):
            list_benchmarks = []

            # Generate QoS for the job
            qos = choices(list(QOS))[0]

            # Generate number of iterations for the job
            num_iters = choices(BatchJob.ITER_CHOICES)[0]

            # Generate benchmarks within the job
            num_benchmarks = self.num_benchmarks_per_job
            if num_benchmarks == -1:
                num_benchmarks = choices(BatchJob.NUM_BENCH_CHOICES)[0]

            for bench_idx in range(num_benchmarks):
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
        # Reverse the sorted results
        indeces = np.argsort(jobs)[::-1]
        jobs = np.sort(jobs)[::-1]
        #print("jobs at the start are :", jobs)
        #print("indeces at the start are :", indeces)
        num_gpus = 0
        f = open("mig_sched.out", "a")

        # returns the index of the element that is to fill the remaining slice
        def best_fill(jobs: list, rem_slice: int) -> (int, int):
            #print("\ninside best_fill, jobs are: ", jobs)
            # find elements that can fill the spot
            # if rem_slice is 4, handle special case
            fours = np.where(jobs == 4)[0].size
            threes = np.where(jobs == 3)[0].size
            twos = np.where(jobs == 2)[0].size
            if (rem_slice == 4) and (fours == 0) and \
            (threes != 0) and (threes % 2 == 0) and \
            (twos + 2 >= (threes / 2)):
                # if we have a slice of size 4 and no more 4s and we have 3s 
                # and we have more than 2 more 2s than twice 3s, we need to 
                # patch this rem_slice of 4 with two 2s instead of 3
                # we return two ints which are the indeces of the two 2s in jobs
                return [np.where(jobs == 2)[0][0], np.where(jobs == 2)[0][1]]

            suitable_indeces = np.where(jobs <= rem_slice) 
            #print("suitable indeces are: ", suitable_indeces)
            suitable_values = np.take(jobs, suitable_indeces)
            #print("suitable values are: ", suitable_values)
            #print("done best_fill\n")
            if suitable_values[0].size != 0:
                return [suitable_indeces[0][0], -1]
            return [-1, -1]

        # fill the machine as much as possible with suitable slices
        def fill_machine(jobs: list, indeces: list, rem_slice: int):
            #print("inside fill machine, rem_slice is ", rem_slice)
            # check if there are suitable jobs in the array and fill
            while np.where(jobs <= rem_slice)[0].size != 0:
                fill_index = best_fill(jobs, rem_slice)
                #print("fill_index is ", fill_index)
                if fill_index[0] == -1:
                    break
                # remove element at fill_index and update jobs and indeces
                job_number = indeces[fill_index][0]
                job_slice = jobs[fill_index][0]
                job_line = str(job_number) + " "
                # if we hit the special case and returned two 2s
                # we need to add the second job to job removal
                if (fill_index[1] != -1):
                    job_number = indeces[fill_index][1]
                    job_slice += jobs[fill_index][1]
                    job_line += str(job_number)
                    indeces = np.delete(indeces, fill_index)
                    jobs = np.delete(jobs, fill_index)
                else:
                    indeces = np.delete(indeces, fill_index[0])
                    jobs = np.delete(jobs, fill_index[0])
                #print("job line is: ", job_line)
                f.write(job_line)
                rem_slice -= job_slice

                #print("remaining slice on this gpu is ", rem_slice)
            #print("jobs array after fill_machine is: ", jobs)
            return jobs, indeces

        job_sizes = "Job sizes   " + np.array_str(jobs) + "\n"
        job_indeces = "Job indeces " + np.array_str(indeces) + "\n" + "\n"
        f.write(job_sizes)
        f.write(job_indeces)

        # Profiling
        start_fill = time.perf_counter()

        # main algo loop where we fill machines with jobs
        while len(jobs) > 0:
            # fill next machine
            num_gpus += 1
            gpu_line = "GPU " + str(num_gpus) + ": Jobs "
            f.write(gpu_line)
            jobs, indeces = fill_machine(jobs, indeces, MAX_PARTITIONS)
            # print("after gpu is filled we have ", jobs, indeces, num_gpus)
            f.write("\n")

        f.close()

        self.time_mig = time.perf_counter() - start_fill

        return num_gpus

    def calculate_gpu_count_gpupool(self, gpupool_config, save=False):
        print("Running GPUPool predictor... ")
        # Profiling
        start_prediction = time.perf_counter()

        # Call two-stage predictor
        self._calculate_gpupool_performance(gpupool_config)

        # Profiling
        self.time_gpupool[gpupool_config.get_time_prediction()] = \
            time.perf_counter() - start_prediction

        print("Running max weight matching solver...")

        # Profiling
        start_matching = time.perf_counter()

        # Call max weight matching solver
        perf_col = gpupool_config.get_perf()
        ws_col = gpupool_config.get_ws()

        f = open("input.js", "w+")
        f.write("var blossom = require('./edmonds-blossom');" + "\n")
        f.write("var data = [" + "\n")

        # iterate over rows and read off the weighted speedups
        for index, row in self.df_pair.iterrows():
            # determine if we include this pair as an edge or not based on if
            # the qos of each job in pair is satisfied
            if ((row['pair_job'].jobs[0].qos.value > row[perf_col].sld[0]) or
                    (row['pair_job'].jobs[1].qos.value > row[perf_col].sld[1])):
                continue
            string_pair = row['pair_str']
            first_job_index = string_pair.split('+')[0].split('-')[1]
            second_job_index = string_pair.split('+')[1].split('-')[1]
            pair_weighted_speedup = row[ws_col]
            input_line = "      [" + str(first_job_index) + ', ' +  \
                    str(second_job_index) + ', ' \
                    + str(round(pair_weighted_speedup, 3)) + '],\n'
            f.write(input_line)
                
        f.write("    ];\n")
        f.write("var results = blossom(data);\n")
        #console.log(util.inspect(array, { maxArrayLength: null }))
        f.write("const util = require('util');\n")
        f.write("console.log(util.inspect(results, {maxArrayLength: null}));\n")
        f.close()

        # print('Resulting input file is:')
        # print('')
        # os.system('cat input.js')
        # print('')

        try:
            matching = subprocess.getoutput("node input.js")
        except subprocess.CalledProcessError as node_except:
            print("GPUPool max match node.js call failed.")
            print("Error code:", node_except.returncode)
            print("Console output:", node_except.output)
            sys.exit(1)

        #matching = subprocess.getoutput("node input.js")
        #print(matching)
        matching = matching.replace(" ", "").replace("]", "").replace("[", "")
        matching = matching.split(",")
        matching = [int(x) for x in matching]
        num_pairs = len([x for x in matching if x >= 0]) // 2
        num_isolated = len([x for x in matching if x == -1])

        os.system('rm input.js')

        # Profiling
        self.time_gpupool[gpupool_config.get_time_matching()] = \
            time.perf_counter() - start_matching

        # Save df_pair
        if save:
            pickle_dir = os.path.join(THIS_DIR, "pickles")
            if not os.path.exists(pickle_dir):
                os.mkdir(pickle_dir)

            self.df_pair.to_pickle(
                os.path.join(
                    pickle_dir,
                    "BatchJob-{}-{}.pkl".format(self.id,
                                                gpupool_config.to_string())))

        return num_pairs + num_isolated

    def calculate_qos_violation_random(self, max_gpu_count):
        # With the same number of GPU that GPUPool uses, how many QoS
        # violations we will get with random assignment
        job_lhs = np.array(self.list_jobs[0:max_gpu_count])
        job_rhs = np.array(self.list_jobs[max_gpu_count:])
        # Remaining jobs in job_lhs will get a full GPU, hence no QoS violations
        job_lhs.resize(job_rhs.size)

        job_pairs = [(job0, job1) for job0, job1 in zip(job_lhs, job_rhs)]

        cores = mp.cpu_count()
        data_split = np.array_split(job_pairs, cores)
        pool = mp.Pool(cores)
        violations = sum(
            pool.map(get_qos_violations, data_split)
        )

        pool.close()
        pool.join()

        return violations





