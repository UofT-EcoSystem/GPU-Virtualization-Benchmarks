import argparse
import subprocess
import pandas as pd
from launch.constant import *

# (Intra CTAs/SM, L2 usage, disable l2D)
bench_opt_config = {
    'cut_sgemm-0': (2, 0.5, False),
    'cut_sgemm-1': (1, 0.25, False),
    'cut_wmma-0': (2, 0.5, False),
    'cut_wmma-1': (2, 0.25, False),
    'parb_cutcp-0': (2, 0.25, False),
    'parb_lbm-0': (3, 0.25, True),
    'parb_spmv-0': (6, 0.25, False),
    'parb_stencil-0': (3, 0.5, False),
}


def parse_args():
    parser = argparse.ArgumentParser("Run app in isolation mode \
            and sweep resource sizes.")

    parser.add_argument('--pair', required=True, nargs='+',
                        help="Apps to run.")
    parser.add_argument('--how', choices=['smk', 'static', 'dynamic'],
                        help='How to partition resources between benchmarks.')
    parser.add_argument('--bench_home', default=DEFAULT_BENCH_HOME,
                        help='Benchmark home folder.')
    parser.add_argument('--no_launch', default=False, action='store_true',
                        help='Do not actually trigger job launching.')
    parser.add_argument('--random', default=False, action='store_true',
                        help='Use random address mapping for global access.')


    results = parser.parse_args()

    return results


app_df = pd.DataFrame.from_dict(bench_opt_config, orient='index',
                                columns=['intra', 'l2', 'bypassl2'])

args = parse_args()

for pair in args.pair:
    apps = pair.split('+')
    all_apps_valid = True
    for app in apps:
        if app not in app_df.index.values:
            print("{0} is not in application map. Skip.".format(pair))
            all_apps_valid = False
            break

    if not all_apps_valid:
        continue

    base_config = "TITANV-SEP_RW-CONCURRENT"
    if args.random:
        base_config += "-RANDOM"

    if args.how == 'smk':
        config_str = base_config
    elif args.how == 'static':
        # intra SM config
        intra_values = [str(app_df.loc[app, 'intra']) for app in apps]
        intra_sm = 'INTRA_0:' + ':'.join(intra_values) + '_CTA'

        # L2 partition config
        # l2_values = [str(app_df.loc[app, 'l2']) for app in apps]
        # FIXME: temp set all L2 partition to 0.5
        l2_values = ['0.5' for app in apps]

        l2 = 'PARTITION_L2_0:' + ':'.join(l2_values)

        config_str = '-'.join([base_config, intra_sm, l2])

        # L2 bypass config
        if app_df.loc[apps[0], 'bypassl2']:
            config_str += "-BYPASS_L2D_S1"
    else:
        # TODO: dynamic sharing policy
        print('Unimplemented error')

    cmd = ['python',
           os.path.join(RUN_HOME, 'run_simulations.py'),
           '-B', pair,
           '-C', config_str,
           '-E', DEFAULT_BENCH_HOME,
           '-N', 'pair'
           ]

    if args.no_launch:
        cmd.append('-n')

    p = subprocess.run(cmd, stdout=subprocess.PIPE)

    print(p.stdout.decode("utf-8"))


