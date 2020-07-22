from random import seed
from random import choices

import data.scripts.common.constants as const


class Job:
    count = 0

    def __init__(self, qos, benchmarks, num_iters):
        self.qos = qos

        # A list of benchmarks
        self.benchmarks = benchmarks

        self.num_iters = num_iters

        self.id = Job.count
        Job.count += 1

        # Hack: append the list to syn_yaml list in const to keep using
        # const
        const.syn_yaml[self.name()] = self.benchmarks

    def name(self):
        return "job-{}".format(self.id)

    def get_seq_cycles(self):
        const.get_seq_cycles(self.name())


class BatchJob:
    REPEAT = 0
    FRESH = 1

    QOS_CHOICES = [0.5, 0.6, 0.7, 0.8, 0.9]
    ITER_CHOICES = [50, 100, 200, 400, 800, 1600]

    count = 0

    def __init__(self, rand_seed, num_benchmarks_per_job, num_jobs,
                 allow_repeat=False, p_repeat=0.2):
        self.rand_seed = rand_seed
        self.num_benchmarks_per_job = num_benchmarks_per_job
        self.num_jobs = num_jobs
        self.allow_repeat = allow_repeat
        self.p_repeat = p_repeat

        self.list_jobs = {}

        self.id = BatchJob.count
        BatchJob.count += 1

    def synthesize_jobs(self):
        self.list_jobs.clear()
        seed(self.rand_seed)

        # Candidates are non multi-kernel benchmarks
        benchmarks = [benchmark for benchmark in const.kernel_yaml if benchmark
                      not in const.multi_kernel_app]

        for app_idx in range(self.num_jobs):
            list_benchmarks = []

            # Generate QoS for the job
            qos = choices(BatchJob.QOS_CHOICES)[0]

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

            new_job = Job(qos, list_benchmarks, num_iters)
            self.list_jobs[new_job.name()] = new_job





