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
from copy import deepcopy

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
                 at_least_once, accuracy_mode, stage2_buffer):
        self.alloc = alloc
        self.stage_1 = stage_1
        self.stage_2 = stage_2
        self.at_least_once = at_least_once
        self.model = None
        self.accuracy_mode = accuracy_mode
        self.stage2_buffer = stage2_buffer

        model_pkl_path = os.path.join(THIS_DIR, "model.pkl")

        from gpupool.predict import RunOption
        if self.stage_1 == StageOne.BoostTree and not accuracy_mode:
            # Runtime mode, simply do inference on a single model
            if os.path.isfile(model_pkl_path):
                print("Load boosting tree from pickle {}"
                      .format(model_pkl_path))

                self.model = pickle.load(open(model_pkl_path, 'rb'))
            else:
                print("Training boosting tree for stage 1.")
                self.model = RunOption.train_boosting_tree(train_all=True)
                pickle.dump(self.model, open(model_pkl_path, 'wb'))

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

    def get_model(self):
        return self.model


class Violation:
    def __init__(self, count=0, err_sum=0, err_max=0, gpu_increase=0,
                 actual_ws=0):
        self.count = count
        self.sum = err_sum
        self.max = err_max
        self.gpu_increase = gpu_increase
        self.actual_ws = actual_ws

    def update(self, actual_qos, target_qos):
        if actual_qos < target_qos:
            self.count += 1
            error = (target_qos - actual_qos) / target_qos
            self.sum += error

            if error > self.max:
                self.max = error

            # return did violate
            return True
        else:
            return False

    def mean_error_pct(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count * 100

    def max_error_pct(self):
        return self.max * 100

    def to_string(self, num_jobs):
        return "{} QoS violations ({:.2f}% jobs, {:.2f}% mean relative " \
               "error, {:.2f}% max error)".format(self.count,
                                                  self.count / num_jobs * 100,
                                                  self.mean_error_pct(),
                                                  self.max_error_pct())

    def __add__(self, other):
        new_count = self.count + other.count
        new_sum = self.sum + other.sum
        new_max = max(self.max, other.max)
        new_gpu_increase = self.gpu_increase + other.gpu_increase
        new_ws = self.actual_ws + other.actual_ws

        return Violation(new_count, new_sum, new_max, new_gpu_increase, new_ws)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


# Top-level functions to be pickled and parallelized on multiple cores
def get_perf(df, config: GpuPoolConfig):
    # Deep copy config so that each process can have a unique model copy to
    # run on if stage1 is boosting tree
    option_copy = deepcopy(config)

    start = time.perf_counter()
    _df_performance = df['pair_job'].apply(
        lambda pair_job:
        pd.Series(pair_job.get_gpupool_performance(option_copy))
    )

    # print("Core:", time.perf_counter() - start, " jobs: ", len(df.index))
    return _df_performance


def get_qos_violations(job_pairs: list):
    violation = Violation()

    for pair in job_pairs:
        qos_goals = [pair[0].qos.value, pair[1].qos.value]

        from gpupool.predict import PairJob
        pair = PairJob([pair[0], pair[1]])
        # print('job names are: ', pair.job_names)
        perf = pair.get_best_effort_performance()
        # print('performance is: ', perf.sld)

        violated_pair = False
        for sld, qos in zip(perf.sld, qos_goals):
            violated_pair |= violation.update(sld, qos)

        if violated_pair:
            violation.gpu_increase += 1
            violation.actual_ws += 2
        else:
            violation.actual_ws += sum(perf.sld)

    return violation


def verify_boosting_tree(df, config: GpuPoolConfig):
    config_copy = deepcopy(config)

    delta = df['pair_job'].apply(
        lambda pair_job: pair_job.verify_boosting_tree(config_copy)
    )

    return delta.tolist()


# End parallel functions


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
                                      mig_ipcs[self.benchmarks[bm_index]][
                                          resources]
            # use restricted runtime to estimate qos at this static partition
            attained_qos = full_runtime / restricted_runtime
            if attained_qos >= self.qos.value:
                return resources + 1, attained_qos

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

    def load_df_from_pickle(self, file_path):
        if os.path.isfile(file_path):
            self.df_pair = pd.read_pickle(file_path)
        else:
            print(file_path, "does not exist. Do nothing.")

    def _calculate_gpupool_performance(self, config: GpuPoolConfig, cores):
        result = parallelize(self.df_pair, cores, get_perf, extra_param=config)
        df_performance = pd.concat(result)

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

    def _max_matching(self, gpupool_config, cores):
        # Call max weight matching solver

        # Profiling
        start_matching = time.perf_counter()

        perf_col = gpupool_config.get_perf()
        ws_col = gpupool_config.get_ws()

        f = open("input.js", "w+")
        f.write("var blossom = require('./edmonds-blossom');" + "\n")
        f.write("var data = [" + "\n")

        # iterate over rows and read off the weighted speedups
        for index, row in self.df_pair.iterrows():
            # determine if we include this pair as an edge or not based on if
            # the qos of each job in pair is satisfied
            jobs = row['pair_job'].jobs
            id_offset = self.list_jobs[0].id
            adjusted_id = [job.id - id_offset for job in jobs]

            # Buffer to tighten the bounds and offset error from stage 1
            buffer = gpupool_config.stage2_buffer
            if (jobs[0].qos.value * (1 + buffer) < row[perf_col].sld[0]) and \
                    (jobs[1].qos.value * (1 + buffer) < row[perf_col].sld[1]):
                input_line = "      [{}, {}, {:.3f}],\n".format(adjusted_id[0],
                                                                adjusted_id[1],
                                                                row[ws_col]
                                                                )
            else:
                input_line = "      [{}, {}, {}],\n".format(adjusted_id[0],
                                                            adjusted_id[1],
                                                            -sys.maxsize - 1)

            f.write(input_line)

        f.write("    ];\n")
        f.write("var results = blossom(data);\n")
        # console.log(util.inspect(array, { maxArrayLength: null }))
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

        # matching = subprocess.getoutput("node input.js")
        # print(matching)
        matching = matching.replace(" ", "").replace("]", "").replace("[", "")
        matching = matching.split(",")
        matching = [int(x) for x in matching]
        num_pairs = len([x for x in matching if x >= 0]) // 2
        num_isolated = len([x for x in matching if x == -1])
        print("number of isolated jobs is ", num_isolated)
        print("number of pairs is ", num_pairs)

        # Profiling
        self.time_gpupool[gpupool_config.get_time_matching()] = \
            time.perf_counter() - start_matching

        # os.system('rm input.js')
        # print(matching)
        # print(self.list_jobs)

        # Parse the output and get qos violations
        print("Running QoS verifications...")
        job_pairs = []
        # print(len(matching))

        for i in range(self.num_jobs):
            if matching[i] < i:
                # either no pair or pair was formed already
                continue
            job_pairs.append((self.list_jobs[matching[i]], self.list_jobs[i]))

        if len(job_pairs) > 0:
            violations_result = parallelize(
                job_pairs, cores, get_qos_violations)
            violation = sum(violations_result)
        else:
            # No jobs are co-running
            violation = Violation()

        num_gpus = num_isolated + num_pairs + violation.gpu_increase
        ws_sum = num_isolated + violation.actual_ws

        return num_gpus, violation, ws_sum / num_gpus

    def boosting_tree_test(self, config: GpuPoolConfig, cores):
        delta = parallelize(self.df_pair, cores, verify_boosting_tree, config)
        # Flatten the array
        import itertools
        delta = list(itertools.chain.from_iterable(delta))

        sum_delta = 0
        len_delta = 0
        for d in delta:
            sum_delta += d[0] * d[1]
            len_delta += d[1]

        average_error = sum_delta / len_delta

        return average_error

    def calculate_gpu_count_mig(self):
        # convert jobs to size slices 
        jobs = [x.calculate_static_partition()[0] for x in self.list_jobs]
        slowdowns = [y.calculate_static_partition()[1] for y in self.list_jobs]
        # Reverse the sorted results
        indeces = np.argsort(jobs)[::-1]
        jobs = np.sort(jobs)[::-1]
        # print("jobs at the start are :", jobs)
        # print("indeces at the start are :", indeces)
        num_gpus = 0
        f = open("mig_sched.out", "a")
        # variable to keep track of which jobs are on same gpu
        paired_jobs = []

        # returns the index of the element that is to fill the remaining slice
        def best_fill(jobs: list, rem_slice: int) -> (int, int):
            # print("\ninside best_fill, jobs are: ", jobs)
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
            # print("suitable indeces are: ", suitable_indeces)
            suitable_values = np.take(jobs, suitable_indeces)
            # print("suitable values are: ", suitable_values)
            # print("done best_fill\n")
            if suitable_values[0].size != 0:
                return [suitable_indeces[0][0], -1]
            return [-1, -1]

        # fill the machine as much as possible with suitable slices
        def fill_machine(jobs: list, indeces: list, rem_slice: int):
            # print("inside fill machine, rem_slice is ", rem_slice)
            # check if there are suitable jobs in the array and fill
            while np.where(jobs <= rem_slice)[0].size != 0:
                fill_index = best_fill(jobs, rem_slice)
                # print("fill_index is ", fill_index)
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
                f.write(job_line)
                # add this job into this gpu's job set
                paired_jobs.append(int(job_line))
                rem_slice -= job_slice

                # print("remaining slice on this gpu is ", rem_slice)
            # print("jobs array after fill_machine is: ", jobs)
            return jobs, indeces

        job_sizes = "Job sizes   " + np.array_str(jobs) + "\n"
        job_indeces = "Job indeces " + np.array_str(indeces) + "\n" + "\n"
        f.write(job_sizes)
        f.write(job_indeces)

        # Profiling
        start_fill = time.perf_counter()

        # main algo loop where we fill machines with jobs
        speedup_sum = 0
        while len(jobs) > 0:
            # fill next machine
            num_gpus += 1
            gpu_line = "GPU " + str(num_gpus) + ": Jobs "
            f.write(gpu_line)
            jobs, indeces = fill_machine(jobs, indeces, MAX_PARTITIONS)
            # add appriate amount to speedup sum based on how many jobs on this 
            # gpu
            if len(paired_jobs) == 1:
                speedup_sum += 1
            else:
                set_slowdown = [slowdowns[job] for job in paired_jobs]
                speedup_sum += sum(set_slowdown)
            # empty the set of jobs on this gpu before we fill next machine
            paired_jobs = []
            # print("after gpu is filled we have ", jobs, indeces, num_gpus)
            f.write("\n")

        f.close()

        self.time_mig = time.perf_counter() - start_fill

        return num_gpus, speedup_sum / num_gpus

    def calculate_gpu_count_gpupool(self, gpupool_config, cores, save=False):
        print("Running GPUPool predictor... ")
        # Profiling
        start_prediction = time.perf_counter()

        # Call two-stage predictor
        self._calculate_gpupool_performance(gpupool_config, cores=cores)

        # Profiling
        self.time_gpupool[gpupool_config.get_time_prediction()] = \
            time.perf_counter() - start_prediction

        print("Running max weight matching solver...")
        gpu_count, violation, ws_total = self._max_matching(gpupool_config,
                                                            cores)

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

        return gpu_count, violation, ws_total

    def calculate_qos_violation_random(self, max_gpu_count, cores):
        # With the same number of GPU that GPUPool uses, how many QoS
        # violations we will get with random assignment
        job_lhs = np.array(self.list_jobs[0:max_gpu_count])
        job_rhs = np.array(self.list_jobs[max_gpu_count:])
        # Remaining jobs in job_lhs will get a full GPU, hence no QoS violations
        job_lhs.resize(job_rhs.size)

        job_pairs = [(job0, job1) for job0, job1 in zip(job_lhs, job_rhs)]

        if len(job_pairs) > 0:
            violations_result = parallelize(
                job_pairs, cores, get_qos_violations)
            violation = sum(violations_result)
        else:
            violation = Violation()

        return violation

    def calculate_qos_viol_dram_bw(self, gpu_avail, cores):

        # access the dram bw of bms:
        df = const.get_pickle('seq.pkl')

        dram_utils = []
        for job in self.list_jobs:
            cycles = job.get_seq_cycles()
            job_total = sum(cycles)
            norm_lengths = [x / job_total for x in cycles]
            avg_dram_bw_util = [float(df[df['pair_str'] == bm]['avg_dram_bw']) for bm
                    in job.benchmarks]
            weighted_arith_mean = sum([norm_lengths[i] * avg_dram_bw_util[i] for i
                    in range(len(norm_lengths))])
            dram_utils.append(weighted_arith_mean)

        # sort the jobs and indeces
        indeces = np.argsort(dram_utils)[::-1]
        job_dram_utils = np.sort(dram_utils)[::-1]
        
        num_singles = -self.num_jobs + 2 * (gpu_avail)
        f = open('dram_bw_based.out', 'a')
        gpu_count = 0
        # allocate singles
        for i in range(num_singles):
            f.write('{}\n'.format(indeces[i]))
            gpu_count += 1

        # pair up rest based on dram_bw
        job_dram_utils = job_dram_utils[num_singles:]
        indeces = indeces[num_singles:]
        job_pair_indeces = []
        while len(indeces) > 1:
            if(len(indeces) >= 2):
                gpu_count += 1
                f.write('{} {}\n'.format(indeces[0], indeces[len(indeces) - 1]))
                job_pair_indeces.append((indeces[0], indeces[len(indeces) - 1]))
                # remove indeces from the list
                indeces = np.delete(indeces, [0, len(indeces) - 1])
                job_dram_utils = np.delete(job_dram_utils, [0, len(job_dram_utils) - 1])
            else:
                f.write('{}\n'.format(indeces[0]))
                # remove indeces from the list
                indeces = np.delete(indeces, 0)
                job_dram_utils = np.delete(job_dram_utils, 0)

        job_pairs = [(self.list_jobs[ind[0]], self.list_jobs[ind[1]]) for ind in
                job_pair_indeces]

        os.system('rm dram_bw_based.out')

        if len(job_pairs) > 0:
            violations_result = parallelize(
                job_pairs, cores, get_qos_violations)
            violation = sum(violations_result)
        else:
            violation = Violation()

        return violation.count
