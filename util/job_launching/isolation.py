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
    parser.add_argument('--intra', action='store_true',
                        help='Run intra experiments or not.')
    parser.add_argument('--inter', action='store_true',
                        help='Run inter experiments or not.')

    parser.add_argument('--cta_configs', default=4, type=int,
                        help='Sweeping step of CTAs/SM for intra-SM sharing.')
    parser.add_argument('--sm_configs', default=4, type=int,
                        help='Sweeping step of SMs for inter-SM sharing.')

    parser.add_argument('--no_launch', default=False, action='store_true',
                        help='Do not actually trigger job launching.')

    parser.add_argument('--env', default='eco', choices=['eco', 'vector'],
                        help='Environment to launch.')

    results = parser.parse_args()

    return results


df_kernel = pd.DataFrame.from_dict(const.kernel_dict, orient='index',
                                   columns=['max_cta', 'grid', 'comp'])
df_kernel['achieved_cta'] = np.minimum(np.ceil(df_kernel['grid'] / 80),
                                       df_kernel['max_cta']).astype('int32')
df_kernel['achieved_sm'] = const.num_sm_volta

args = parse_args()

if args.apps[0] == 'all':
    all_apps = list(const.kernel_dict.keys())

    last_index = min(len(all_apps), args.count + args.id_start)
    args.apps = all_apps[args.id_start:last_index]

for app in args.apps:
    if app in const.multi_kernel_app:
        kernels = ["{0}:{1}".format(app, kidx)
                   for kidx in range(1, const.multi_kernel_app[app]+1)]
        launch_name = 'isolation-multi'
    elif app in df_kernel.index.values:
        kernels = [app]
        launch_name = 'isolation'
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
            kidx = split_kernel[1]
            configs = ["-".join([cfg, "MIX_{}_KIDX".format(kidx)])
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

    if args.intra:
        for k in kernels:
            _abs_max = df_kernel.loc[k, 'achieved_cta']
            _step = max(1, _abs_max // args.cta_configs)

            intra_sm = ["INTRA_0:{0}:0_CTA".format(i)
                        for i in range(_step, _abs_max, _step)]
            intra_sm.append("INTRA_0:{0}:0_CTA".format(_abs_max))

            launch_job(intra_sm, launch_name + '-intra', k)

    if args.inter:
        for k in kernels:
            _abs_max = df_kernel.loc[k, 'achieved_sm']
            _step = max(1, _abs_max // args.sm_configs)

            inter_sm = ["INTER_0:{0}:0_SM".format(i)
                        for i in range(_step, _abs_max, _step)]
            inter_sm.append("INTER_0:{0}:0_SM".format(_abs_max))

            print(app, inter_sm)

            launch_job(inter_sm, launch_name + '-inter', k)
