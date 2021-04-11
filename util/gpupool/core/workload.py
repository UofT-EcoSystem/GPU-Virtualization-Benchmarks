from random import seed
from random import choices
import pandas as pd
import pickle
import os
import subprocess
import numpy as np
import time
import sys
import math

import data.scripts.common.constants as const
from gpupool.core.helper import QOS, GpuPoolConfig, Violation
from gpupool.core.configs import *
from gpupool.core.parallelize import *

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


class Job:
    count = 0
    job_list = {}

    def __init__(self, qos: QOS, num_iters, benchmarks: list):
        self.qos = qos
        self.num_iters = num_iters

        self.sld_gpupool = 0
        self.sld_heuristic = 0
        self.sld_mig = 0

        self.bw = -1
        self.vector = []

        # Create new job with a list of benchmarks
        self.benchmarks = benchmarks

        # Create metric table from df_intra
        df_intra = const.get_pickle('intra.pkl')
        # Grab the used benchmarks
        df_benchmarks = df_intra[df_intra['pair_str'].isin(benchmarks)]
        # Shave off unused cols
        df_benchmarks = df_benchmarks[KERNEL_COLS]
        # Get rid of the ones with norm_ipc below 0.5
        df_benchmarks = df_benchmarks[df_benchmarks['norm_ipc'] > 0.5]
        # assign kernel ids
        df_benchmarks['kernel_id'] = df_benchmarks['pair_str'].apply(
            lambda pair_str: self.benchmarks.index(pair_str)
        )
        assert not df_benchmarks['kernel_id'].isnull().values.any()
        self.df_benchmarks = df_benchmarks.astype({'intra': 'int32'})

        self.id = Job.count
        Job.count += 1

        self.name = "job-{}".format(self.id)

        # FIXME: why is this needed
        Job.job_list[self.name] = self.benchmarks

    def get_seq_cycles(self):
        result = [const.get_seq_cycles(bench)[0] for bench in self.benchmarks]
        return result

    def get_num_benchmarks(self):
        return len(self.benchmarks)

    # FIXME: store this value in obj instead?
    def calculate_static_partition(self):
        mig_pickle_path = os.path.join(const.DATA_HOME, 'pickles/mig_ipcs')
        mig_ipcs = pickle.load(open(mig_pickle_path, 'rb'))
        cycle_lengths = self.get_seq_cycles()

        full_runtime = sum(cycle_lengths)
        # for resources in range(MAX_PARTITIONS):
        for resources in MIG_PARTITIONS:
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
                self.sld_mig = attained_qos
                return resources + 1, attained_qos

        return MAX_PARTITIONS, 1

    def calculate_metric_vector(self):
        if hasattr(self, 'vector') and len(self.vector) > 0:
            return

        cycles = self.get_seq_cycles()
        metrics = ['ipc', 'avg_mem_lat', 'mpc', 'avg_dram_bw',
                   'thread_count', 'sp_busy', 'int_busy',
                   'not_selected_cycles', 'l2_miss_rate']

        df = const.get_pickle('seq.pkl')

        df['mpc'] = df['mem_count'] / df['runtime']
        df['thread_count'] = df.apply(
            lambda row: const.get_achieved_cta(row['pair_str']) *
                        const.get_block_size(row['pair_str']),
            axis=1
        )

        job_total = sum(cycles)
        norm_lengths = [x / job_total for x in cycles]

        self.vector = []
        for metric in metrics:
            utilization = [float(df[df['pair_str'] == bm][metric])
                           for bm in self.benchmarks]
            weighted_arith_mean = sum([norm_lengths[i] * utilization[i]
                                       for i in range(len(norm_lengths))])
            self.vector.append(weighted_arith_mean)

    def similar(self, other):
        # calculate cosine similarity between two jobs
        def square_rooted(x):
            return round(math.sqrt(sum([a * a for a in x])), 3)

        def cosine_similarity(x, y):
            numerator = sum(a * b for a, b in zip(x, y))
            denominator = square_rooted(x) * square_rooted(y)
            return round(numerator / float(denominator), 3)

        self.calculate_metric_vector()
        other.calculate_metric_vector()

        return cosine_similarity(self.vector, other.vector)


class BatchJob:
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

        # Profiling info in seconds
        # self.time_gpupool = {}
        self.time_mig = 0

    def _two_stage_predict(self, config: GpuPoolConfig, cores):
        # Shuffle df_pair for load balancing
        self.df_pair = self.df_pair.sample(frac=1).reset_index(drop=True)

        # Profiling
        start_prediction = time.perf_counter()

        result = parallelize(self.df_pair, cores, two_stage_predict,
                             extra_param=config)

        # End profiling
        total_latency = time.perf_counter() - start_prediction

        # Collect stage1 and stage2 latency
        perf_col = config.get_perf()
        stage1_latency = [np.sum(r[perf_col].apply(lambda p: p.time_stage1))
                          for r in result]
        stage2_latency = [np.sum(r[perf_col].apply(lambda p: p.time_stage2))
                          for r in result]

        latency = {
            'total': total_latency,
            'stage1': stage1_latency,
            'stage2': stage2_latency
        }

        df_performance = pd.concat(result)

        # FIXME: wtf is this?
        # Drop the same columns
        drop_col = [col for col in self.df_pair.columns
                    if col in df_performance.columns]
        self.df_pair.drop(columns=drop_col, inplace=True)

        self.df_pair = self.df_pair.merge(df_performance,
                                          left_index=True, right_index=True)

        # df_pair is not sorted by pair_str in each run
        # which resulted in non-reproducible output each time
        self.df_pair = self.df_pair.sort_values('pair_str').reset_index()

        return latency

    def _synthesize_jobs(self):
        self.list_jobs.clear()
        seed(self.rand_seed)

        # Candidates are non multi-kernel benchmarks
        benchmarks = [benchmark for benchmark in const.kernel_yaml if benchmark
                      not in const.multi_kernel_app]

        if self.num_jobs == -1:
            self.num_jobs = choices(NUM_JOB_CHOICES)[0]

        print("BatchJob {} synthesizes {} jobs.".format(self.id, self.num_jobs))

        for app_idx in range(self.num_jobs):
            list_benchmarks = []

            # Generate QoS for the job
            qos = choices(list(QOS))[0]

            # Generate number of iterations for the job
            num_iters = choices(ITER_CHOICES)[0]

            # Generate benchmarks within the job
            num_benchmarks = self.num_benchmarks_per_job
            if num_benchmarks == -1:
                num_benchmarks = choices(NUM_BENCH_CHOICES)[0]

            for bench_idx in range(num_benchmarks):
                if bench_idx == 0:
                    # The first one cannot be repeat
                    list_benchmarks += choices(benchmarks)
                else:
                    # Figure out if we are repeating or selecting
                    # a fresh benchmark
                    repeat_or_fresh = choices([REPEAT, FRESH],
                                              weights=[self.p_repeat,
                                                       1 - self.p_repeat])[0]
                    if self.allow_repeat and repeat_or_fresh == REPEAT:
                        list_benchmarks.append('repeat')
                    else:
                        # Make sure we don't pick the same benchmark again
                        leftover = [benchmark for benchmark in benchmarks
                                    if benchmark not in list_benchmarks]
                        list_benchmarks += choices(leftover)

            new_job = Job(qos, num_iters, benchmarks=list_benchmarks)
            self.list_jobs.append(new_job)

    def _create_pairs(self):
        from gpupool.core.predict import PairJob
        pairs = [PairJob([lhs, rhs])
                 for lhs in self.list_jobs for rhs in self.list_jobs
                 if lhs.name < rhs.name]

        self.df_pair['pair_job'] = pairs
        self.df_pair['pair_str'] = self.df_pair['pair_job'].apply(lambda x:
                                                                  x.name())

    # FIXME
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

    def _max_matching(self, stage2_buffer=0, gpupool=True, perf_col=None):

        f = open("input.js", "w+")
        f.write("var blossom = require('./edmonds-blossom');" + "\n")
        f.write("var data = [" + "\n")

        for index, row in self.df_pair.iterrows():
            # determine if we include this pair as an edge or not based on if
            # the qos of each job in pair is satisfied
            jobs = row['pair_job'].jobs

            if gpupool:
                # Buffer to tighten the bounds and offset error from stage 1
                buffer = stage2_buffer

                target = [job.qos.value * (1 + buffer) for job in jobs]
                prediction = row[perf_col].sld
                condition = (prediction[0] > target[0]) and \
                            (prediction[1] > target[1])
            else:
                threshold = 0.8
                condition = row['similar'] < threshold

            if condition:
                input_line = "      [{}, {}, {}],\n".format(jobs[0].id,
                                                            jobs[1].id,
                                                            1
                                                            )
            else:
                input_line = "      [{}, {}, {}],\n".format(jobs[0].id,
                                                            jobs[1].id,
                                                            -sys.maxsize + 1)

            f.write(input_line)

        f.write("    ];\n")
        f.write("var results = blossom(data);\n")
        # console.log(util.inspect(array, { maxArrayLength: null }))
        f.write("const util = require('util');\n")
        f.write("console.log(util.inspect(results, {maxArrayLength: null}));\n")
        f.close()

        try:
            matching = subprocess.getoutput("node input.js")
        except subprocess.CalledProcessError as node_except:
            print("GPUPool max match node.js call failed.")
            print("Error code:", node_except.returncode)
            print("Console output:", node_except.output)
            sys.exit(1)

        matching = matching.replace(" ", "").replace("]", "").replace("[", "")
        matching = matching.split(",")
        matching = [int(x) for x in matching]

        list_pair_str = []
        for i in range(len(self.list_jobs)):
            if matching[i] < i:
                # either no pair or pair was formed already
                continue

            job_pair = [self.list_jobs[matching[i]].name,
                        self.list_jobs[i].name]
            job_pair.sort()
            pair_str = "+".join(job_pair)
            list_pair_str.append(pair_str)

        # Clean up temp file
        os.system('rm input.js')

        print(list_pair_str)

        return list_pair_str

    def _verify_qos(self, list_pair_str, gpupool_config=None, cores=32):
        if len(list_pair_str) == 0:
            return Violation()

        # Parse the output and get qos violations
        print("Running QoS verifications...")
        df_selected = self.df_pair[self.df_pair['pair_str'].isin(list_pair_str)]
        options = df_selected[gpupool_config.get_option()]
        assert len(options) == len(list_pair_str)

        violations_result = parallelize(options, cores, verify_qos)

        # Aggregate all the violation results
        violation = sum(violations_result)

        return violation

    # Calculate GPU count for different systems: #
    # MIG, GPUPool, Random, DRAM, Similarity #
    def calculate_gpu_count_mig(self):
        print("Running MIG job scheduling...")
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
        ws_list = []
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
                ws_list.append(1)
            else:
                set_slowdown = [slowdowns[job] for job in paired_jobs]
                speedup_sum += sum(set_slowdown)
                ws_list.append(sum(set_slowdown))
            # empty the set of jobs on this gpu before we fill next machine
            paired_jobs = []
            # print("after gpu is filled we have ", jobs, indeces, num_gpus)
            f.write("\n")

        f.close()

        self.time_mig = time.perf_counter() - start_fill

        print("Done with MIG scheduling.")

        result = {
            "gpu_count": num_gpus,
            "ws_total": speedup_sum / num_gpus,
            "ws_list": ws_list
        }

        return result

    def calculate_gpu_count_gpupool(self, gpupool_config, cores, save=False):
        # Call two-stage predictor #
        print("Running GPUPool predictor... ")
        pred_latency = self._two_stage_predict(gpupool_config, cores=cores)

        # We do a sweep to search for good QoS margin:
        margin_min = 0.04
        margin_step = 0.02
        margin_max = 2
        margin_selected = margin_max
        list_pair_str = []
        violation = Violation()
        for margin in np.arange(margin_min, margin_max, margin_step):
            margin_selected = margin

            # Call max match job dispatcher #
            print("Running max weight matching solver...")
            start_matching = time.perf_counter()

            perf_col = gpupool_config.get_perf()
            list_pair_str = self._max_matching(gpupool=True,
                                               stage2_buffer=margin,
                                               perf_col=perf_col)

            pred_latency['matching'] = time.perf_counter() - start_matching

            # Verify QoS #
            violation = self._verify_qos(list_pair_str, gpupool_config,
                                         cores=cores)

            if violation.gpu_increase == 0:
                break

        # Calculate final metrics #
        pairs_predicted = len(list_pair_str)
        isolated_predicted = len(self.list_jobs) - 2 * pairs_predicted
        num_gpus = isolated_predicted + pairs_predicted + violation.gpu_increase

        ws_avg = (isolated_predicted + violation.actual_ws) / num_gpus
        ws_list = [1] * isolated_predicted + violation.actual_ws_list

        # Save df_pair #
        if save:
            pickle_dir = os.path.join(THIS_DIR, "pickles")
            if not os.path.exists(pickle_dir):
                os.mkdir(pickle_dir)

            self.df_pair.to_pickle(
                os.path.join(
                    pickle_dir,
                    "BatchJob-{}-{}.pkl".format(self.id,
                                                gpupool_config.to_string())))

        print("Done with GPUPool Calculations.\n")

        result = {'gpu_count': num_gpus,
                  'margin': margin_selected,
                  'violation': violation,
                  'ws_total': ws_avg,
                  'ws_list': ws_list,
                  'pred_latency': pred_latency,
                  }

        return result

    def calculate_qos_violation_random(self, max_gpu_count, cores):
        print("Running random matching...")

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

        print("Done random matching.\n")
        return violation

    def calculate_qos_viol_dram_bw(self, gpu_avail, cores):
        print("Running heuristic-based scheduling...")

        # access the dram bw of bms:
        df = const.get_pickle('seq.pkl')

        for job in self.list_jobs:
            cycles = job.get_seq_cycles()
            job_total = sum(cycles)
            norm_lengths = [x / job_total for x in cycles]
            avg_dram_bw_util = [float(df[df['pair_str'] == bm]['avg_dram_bw'])
                                for bm in job.benchmarks]
            weighted_arith_mean = sum([norm_lengths[i] * avg_dram_bw_util[i]
                                       for i in range(len(norm_lengths))])

            job.bw = weighted_arith_mean

        # Sort in descending order
        sorted_jobs = sorted(self.list_jobs, key=lambda x: x.bw,
                             reverse=True)

        num_singles = int(-self.num_jobs + 2 * gpu_avail)
        shared_jobs = sorted_jobs[num_singles:]

        job_pairs = []
        while len(shared_jobs) > 2:
            job_pairs.append((shared_jobs[0], shared_jobs[-1]))
            shared_jobs = shared_jobs[1:-1]

        if len(job_pairs) > 0:
            violations_result = parallelize(
                job_pairs, cores, get_qos_violations)
            violation = sum(violations_result)
        else:
            violation = Violation()

        ws_list = violation.actual_ws_list + [1] * num_singles
        gpu_count = gpu_avail + violation.gpu_increase

        result = {
            "gpu_count": gpu_count,
            "violation": violation,
            "ws_list": ws_list,
            "ws_total": np.average(ws_list)
        }
        return result

    # FIXME: updated max_match function call
    def calculate_qos_viol_similarity(self, cores):
        print("Running similarity-based scheduling...")

        self.df_pair['similar'] = self.df_pair.apply(
            lambda row: row['pair_job'].jobs[0].similar(row['pair_job'].jobs[1]),
            axis=1
        )

        matching, num_pairs, num_isolated = self._max_matching(gpupool=False)

        print("job pairs", num_pairs)
        print("num singles", num_isolated)

        job_pairs = []
        for i in range(len(self.list_jobs)):
            if matching[i] < i:
                # either no pair or pair was formed already
                continue

            job_pairs.append((self.list_jobs[matching[i]],
                              self.list_jobs[i]))

        if len(job_pairs) > 0:
            violations_result = parallelize(
                job_pairs, cores, get_qos_violations)
            violation = sum(violations_result)
        else:
            violation = Violation()

        ws_list = violation.actual_ws_list + [1] * num_isolated
        gpu_migrated_count = len(job_pairs) + violation.gpu_increase \
                             + num_isolated

        return violation, ws_list, gpu_migrated_count

    # deprecated
    def load_df_from_pickle(self, file_path):
        if os.path.isfile(file_path):
            self.df_pair = pd.read_pickle(file_path)
        else:
            print(file_path, "does not exist. Do nothing.")

