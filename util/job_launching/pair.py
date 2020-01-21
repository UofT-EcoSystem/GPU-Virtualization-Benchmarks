import argparse
import subprocess
import pandas as pd
from job_launching.constant import *
import data.scripts.common.constants as const
import data.scripts.gen_tables.gen_pair_configs as dynamic

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
    parser.add_argument('--app_match', default='',
                        help='Select all pairs that include this app. Only '
                             'checked when all is passed to --pair.')
    parser.add_argument('--how', choices=['smk', 'static', 'dynamic'],
                        help='How to partition resources between benchmarks.')
    parser.add_argument('--bench_home', default=DEFAULT_BENCH_HOME,
                        help='Benchmark home folder.')
    parser.add_argument('--no_launch', default=False, action='store_true',
                        help='Do not actually trigger job launching.')
    parser.add_argument('--random', default=False, action='store_true',
                        help='Use random address mapping for global access.')
    parser.add_argument('--id_start', type=int, default=0,
                        help='For all pairs only. Starting pair id.')
    parser.add_argument('--count', type=int, default=20,
                        help='Max number of simulations to launch.')
    parser.add_argument('--env', default='eco', choices=['eco', 'vector'],
                        help='Environment to launch.')
    parser.add_argument('--cap',
                        type=float,
                        default=2.5,
                        help='Fail fast simulation: cap runtime at n times '
                             'the longer kernel. Default is 2.5x.')

    results = parser.parse_args()

    return results


app_df = pd.DataFrame.from_dict(bench_opt_config, orient='index',
                                columns=['intra', 'l2', 'bypassl2'])

args = parse_args()

# Determine what app pairs to launch
if args.pair[0] == 'all':
    pairs = []
    for bench0 in const.app_for_pair:
        for bench1 in const.app_for_pair:
            if bench0 < bench1:
                pairs.append('+'.join([bench0, bench1]))

    if not args.id_start < len(pairs):
        print('Length of all pairs is {0} but id_start is {1}'
              .format(len(pairs), args.id_start))
        exit(1)
    id_end = args.id_start + args.count

    if id_end > len(pairs):
        id_end = len(pairs)

    args.pair = pairs[args.id_start:id_end]

    # Drop pairs that do not include app match
    if args.app_match != '':
        args.pair = [p for p in args.pair if args.app_match in p]

# Keep track of total jobs launched
job_count = 0
pair_count = 0

for pair in args.pair:
    apps = pair.split('+')

    base_config = "TITANV-SEP_RW-CONCURRENT"
    if args.random:
        base_config += "-RANDOM"

    if args.how == 'smk':
        configs = [base_config]
    elif args.how == 'static':
        # Check if our table has the right info for apps
        all_apps_valid = True
        for app in apps:
            if app not in app_df.index.values:
                print("{0} is not in application map. Skip.".format(pair))
                all_apps_valid = False
                break
        if not all_apps_valid:
            continue

        # intra SM config
        intra_values = [str(app_df.loc[app, 'intra']) for app in apps]
        intra_sm = 'INTRA_0:' + ':'.join(intra_values) + '_CTA'

        # L2 partition config
        # l2_values = [str(app_df.loc[app, 'l2']) for app in apps]
        # FIXME: temp set all L2 partition to 0.5
        l2_values = ['0.5' for app in apps]

        l2 = 'PARTITION_L2_0:' + ':'.join(l2_values)

        config = '-'.join([base_config, intra_sm, l2])

        # L2 bypass config
        if app_df.loc[apps[0], 'bypassl2']:
            config += "-BYPASS_L2D_S1"

        configs = [config]
    else:
        pair_config_args = ['--apps'] + apps

        if args.random:
            pair_config_args.append('--random')

        pair_config_args.append('--print')

        pair_config_args += ['--cap', str(args.cap)]

        # dynamic.main returns an array of candidate configs
        configs = dynamic.main(pair_config_args)

        if len(configs) == 0:
            # gen_pair_configs did not generate feasible config candidates
            continue

    for config in configs:
        cmd = ['python3',
               os.path.join(RUN_HOME, 'run_simulations.py'),
               '-B', pair,
               '-C', config,
               '-E', args.bench_home,
               '-N', 'pair-' + args.how,
               '--env', args.env,
               ]

        if args.no_launch:
            cmd.append('-n')

        p = subprocess.run(cmd, stdout=subprocess.PIPE)
        print(p.stdout.decode("utf-8"))

    pair_count += 1
    job_count += len(configs)

print('\n')
print('>'*10, 'Summary', '<'*10)
print('Total app pairs considered:', pair_count)
print('Total jobs launched:', job_count)
