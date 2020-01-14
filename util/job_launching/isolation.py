import argparse
import subprocess
import os
import pandas as pd
import numpy as np
import sys
from launch.constant import *


def parse_args():
    parser = argparse.ArgumentParser("Run app in isolation mode \
            and sweep resource sizes.")

    parser.add_argument('--apps', required=True, nargs='+', 
            help="Apps to run.")
    parser.add_argument('--bench_home', default=DEFAULT_BENCH_HOME, 
            help='Benchmark home folder.')
    parser.add_argument('--intra', default=True, action='store_true',
            help='Run intra experiments or not. Default to yes.')
    parser.add_argument('--inter', default=False, action='store_true',
            help='Run inter experiments or not. Default to no.')

    parser.add_argument('--cta_configs', default=4, type=int, 
            help='Sweeping step of CTAs/SM for intra-SM sharing.')
    parser.add_argument('--sm_configs', default=4, type=int, 
            help='Sweeping step of SMs for inter-SM sharing.')

    parser.add_argument('--no_launch', default=False, action='store_true',
            help='Do not actually trigger job launching.')

    results = parser.parse_args()

    return results


app_df = pd.DataFrame.from_dict(app_dict, orient='index',
                                columns=['max_cta', 'grid'])
app_df['achieved_cta'] = pd.DataFrame([np.ceil(app_df['grid'] / 80),
                                       app_df['max_cta']]).min().astype('int32')
app_df['achieved_sm'] = np.ceil(app_df['grid'] / app_df['max_cta']) \
    .astype('int32')

args = parse_args()

for app in args.apps:
    if app not in app_df.index.values:
        print("{0} is not in application map. Skip.".format(app))

    def launch_job(sm_config, jobname):
        configs = ["-".join([base_config, sm, l2])
                   for sm in sm_config for l2 in l2_partition]

#        if app in mem_intense:
#            configs = configs + [c + '-' + bypass_l2d for c in configs \
#                    if ('L2_0:0.125' in c) or ('L2_0:0.25' in c)]

        config_str = ','.join(configs)

        cmd =  ['python',
                os.path.join(RUN_HOME, 'run_simulations.py'),
                '-B', app,
                '-C', config_str,
                '-E', DEFAULT_BENCH_HOME,
                '-N', jobname
                ]

        if args.no_launch:
            cmd.append('-n')

        p = subprocess.run(cmd, stdout=subprocess.PIPE)

        print(p.stdout.decode("utf-8"))

    base_config = "TITANV-SEP_RW-CONCURRENT"
    bypass_l2d = "BYPASS_L2D_S1"

    l2_fract = [0.25, 0.5, 0.75, 1.0]
    l2_partition = ["PARTITION_L2_0:{0}:{1}".format(f, 1 - f) for f in l2_fract]

    if args.intra:
        _abs_max = app_df.loc[app, 'achieved_cta']
        _step = max(1, _abs_max // args.cta_configs)

        intra_sm = ["INTRA_0:{0}:0_CTA".format(i) 
                for i in range(_step, _abs_max, _step)]
        intra_sm.append("INTRA_0:{0}:0_CTA".format(_abs_max))

        launch_job(intra_sm, 'isolation-intra')

    if args.inter:
        _abs_max = app_df.loc[app, 'achieved_sm']
        _step = max(1, _abs_max // args.sm_configs)

        inter_sm = ["INTER_0:{0}:0_SM".format(i) 
                for i in range(_step, _abs_max, _step)]
        inter_sm.append("INTER_0:{0}:0_SM".format(_abs_max))

        launch_job(inter_sm, 'isolation-inter')


