import argparse
import subprocess
import pandas as pd
import numpy as np
from job_launching.constant import *
import data.scripts.common.constants as const


def parse_args():
    parser = argparse.ArgumentParser("Run app in isolation mode \
            and sweep resource sizes.")

    parser.add_argument('--apps', required=True, nargs='+',
                        help="Apps to run.")
    parser.add_argument('--id_start', type=int, default=0,
                        help='For all apps only. Starting app id.')
    parser.add_argument('--count', type=int, default=20,
                        help='Max number of apps to launch.')
    parser.add_argument('--bench_home', default=DEFAULT_BENCH_HOME,
                        help='Benchmark home folder.')
    parser.add_argument('--mem_steps', default=4, type=int,
                        help='Number of steps in the sweep for memory.')
    parser.add_argument('--no_launch', default=False, action='store_true',
                        help='Do not actually trigger job launching.')

    parser.add_argument('--env', default='eco', choices=['eco', 'vector'],
                        help='Environment to launch.')

    results = parser.parse_args()

    return results


mem_ch = const.num_mem_ch_volta // 2

args = parse_args()

if args.apps[0] == 'all':
    all_apps = list(const.kernel_yaml.keys())

    last_index = min(len(all_apps), args.count + args.id_start)
    args.apps = all_apps[args.id_start:last_index]

for app in args.apps:
    if app in const.multi_kernel_app.keys():
        kernels = ["{0}:{1}".format(app, kidx)
                   for kidx in const.kernel_yaml[app].keys()]
    elif app in const.kernel_yaml.keys():
        kernels = [app]
    else:
        print("{0} is not in application map. Skip.".format(app))
        continue


    def launch_job(sm_config, jobname, kernel):
        #        configs = ["-".join([base_config, sm, l2])
        #                   for sm in sm_config for l2 in l2_partition]
        configs = ["-".join([base_config, sm]) for sm in sm_config]

        split_kernel = kernel.split(':')
        if len(split_kernel) > 1:
            # This is a kernel part of a multi-kernel benchmark
            # append skip kidx config to do performance simulation only on
            # this kernel
            jobname += '-multi'
            bench = split_kernel[0]
            kidx = split_kernel[1]
            num_kernel = const.multi_kernel_app[bench]
            configs = ["-".join([cfg,
                                 "MIX_0:{}:0_KIDX".format(kidx),
                                 "NUM_0:{}:0_KERNEL".format(num_kernel)]
                                )
                       for cfg in configs]

        cmd = ['python3',
               os.path.join(RUN_HOME, 'run_simulations.py'),
               '--app', split_kernel[0],
               '--bench_home', args.bench_home,
               '--launch_name', jobname,
               '--env', args.env
               ]

        cmd += ['--config'] + configs

        if args.no_launch:
            cmd.append('--no_launch')

        p = subprocess.run(cmd, stdout=subprocess.PIPE)

        print(p.stdout.decode("utf-8"))


    base_config = "TITANV-SEP_RW-PAE-CONCURRENT"


    for k in kernels:
        _step = max(1, mem_ch // args.mem_steps)

        mem_channels = ["MIG_{0}_MEM".format(i)
                    for i in range(_step, mem_ch, _step)]
        mem_channels.append("MIG_{0}_MEM".format(mem_ch))

        print(app, mem_channels)

        launch_job(mem_channels, 'mig_mem_sweep', k)
