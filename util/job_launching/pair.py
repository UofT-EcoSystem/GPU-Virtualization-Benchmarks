import argparse
import subprocess
import pandas as pd
import numpy as np
import sys
from job_launching.constant import *
import data.scripts.common.constants as const
import data.scripts.common.help_iso as help_iso
import data.scripts.gen_tables.gen_pair_configs as dynamic
import data.scripts.gen_tables.search_best_inter as inter

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

args = None


def parse_args():
    parser = argparse.ArgumentParser("Run app in isolation mode \
            and sweep resource sizes.")

    parser.add_argument('--pair', required=True, nargs='+',
                        help="Apps to run.")
    parser.add_argument('--id_start', type=int, default=0,
                        help='For all pairs only. Starting pair id.')
    parser.add_argument('--count', type=int, default=20,
                        help='Max number of simulations to launch.')

    parser.add_argument('--app_match', default='',
                        help='Select all pairs that include this app. Only '
                             'checked when all is passed to --pair.')
    parser.add_argument('--app_exclude', default=[], nargs='+',
                        help='Select all pairs that do not include this app. '
                             'Only checked when all is passed to --pair.')

    parser.add_argument('--how', choices=['smk', 'static', 'dynamic',
                                          'inter', 'ctx'],
                        help='How to partition resources between benchmarks.')
    parser.add_argument('--num_slice', default=4, type=int,
                        help='If how is ctx, num_slice specifies how many '
                             'slice configs to sweep.')
    parser.add_argument('--intra_pkl',
                        default=os.path.join(const.DATA_HOME, 'pkl/intra.pkl'),
                        help='If how is dynamic, path to the intra pickle '
                             'file.')
    parser.add_argument('--top',
                        action='store_true',
                        help='If how is dynamic, only select top candidates.')
    parser.add_argument('--cap',
                        type=float,
                        default=2.5,
                        help='Fail fast simulation: cap runtime at n times '
                             'the longer kernel. Default is 2.5x.')

    parser.add_argument('--bench_home', default=DEFAULT_BENCH_HOME,
                        help='Benchmark home folder.')
    parser.add_argument('--no_launch', default=False, action='store_true',
                        help='Do not actually trigger job launching.')

    parser.add_argument('--env', default='eco', choices=['eco', 'vector'],
                        help='Environment to launch.')

    parser.add_argument('--new_only', action='store_true',
                        help='If this flag is passed, do not launch jobs if '
                             'the same job config exists. Run folder is '
                             'checked.')
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing sim run dir completely.")

    global args
    args = parser.parse_args()


def launch_job(*configs, pair, multi=False):
    launch_name = 'pair-' + args.how
    if multi:
        launch_name += '-multi'

    for config in configs:
        cmd = ['python3',
               os.path.join(RUN_HOME, 'run_simulations.py'),
               '--app', pair,
               '--config', config,
               '--bench_home', args.bench_home,
               '--launch_name', launch_name,
               '--env', args.env,
               ]

        if args.no_launch:
            cmd.append('--no_launch')

        if args.new_only:
            cmd.append('--new_only')

        if args.overwrite:
            cmd.append('--overwrite')

        p = subprocess.run(cmd, stdout=subprocess.PIPE)

        if not args.no_launch:
            print(p.stdout.decode("utf-8"))

    print('Launching pair: ', pair)
    [print(cfg) for cfg in configs]


def process_smk(pair, config):
    launch_job(config, pair=pair)
    return 1


def process_static(pair, base_config):
    apps = pair.split('+')
    app_df = pd.DataFrame.from_dict(bench_opt_config, orient='index',
                                    columns=['intra', 'l2', 'bypassl2'])

    # Check if our table has the right info for apps
    all_apps_valid = True
    for app in apps:
        if app not in app_df.index.values:
            print("{0} is not in application map. Skip.".format(apps))
            all_apps_valid = False
            break
    if not all_apps_valid:
        return []

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

    launch_job(*configs, pair=pair)

    return len(configs)


def process_dynamic(pair):
    # This simulation simulates pairs of single kernels
    apps = pair.split('+')
    pair_config_args = ['--apps'] + apps

    pair_config_args += ['--cap', str(args.cap)]
    pair_config_args += ['--intra_pkl', args.intra_pkl]

    if args.top:
        pair_config_args += ['--top']

    # dynamic.main returns an array of candidate configs
    configs = dynamic.main(pair_config_args)

    split_kernels = [k.split(':') for k in apps]
    kidx = [sk[1] if len(sk) > 1 else 1 for sk in split_kernels]

    configs = ["-".join([cfg, "MIX_0:{0}:{1}_KIDX".format(kidx[0], kidx[1])])
               for cfg in configs]

    # Each app in pair should indicate which kernel is run for multi-kernel
    # benchmark
    pair = '+'.join([split_kernels[0][0], split_kernels[1][0]])

    multi = any(len(sk) > 1 for sk in split_kernels)

    launch_job(*configs, pair=pair, multi=multi)

    return len(configs)


def process_inter(pair):
    apps = pair.split('+')
    # inter-SM sharing
    pair_config_args = ['--apps'] + apps
    pair_config_args.append('--print')
    pair_config_args += ['--cap', str(args.cap)]
    pair_config_args += ['--how', 'local']

    configs = inter.main(pair_config_args)

    launch_job(*configs, pair=pair)

    return len(configs)


def process_ctx(pair, base_config, df_seq_multi):
    print(df_seq_multi.index)
    apps = pair.split('+')

    step = 1.0 / args.num_slice
    configs = [base_config + '-INTRA_0:{0}:{1}_RATIO'.format(r, 1 - r) for r in
               np.arange(step, 1.0 + step, step)]

    max_cycles = 0

    for app in apps:
        # check how many kernels there are in the app
        num_kernels = const.multi_kernel_app[app]
        sum_cycles = 0
        for kidx in range(1, num_kernels + 1, 1):
            sum_cycles += df_seq_multi.loc[(app, kidx)]['runtime']

        if sum_cycles > max_cycles:
            max_cycles = sum_cycles

    cap_cycles = args.cap * max_cycles
    cap_config = '-CAP_{0}_CYCLE'.format(int(cap_cycles))

    configs = [c + cap_config for c in configs]

    launch_job(*configs, pair=pair)

    return len(configs)


def process_pairs():
    global args
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

        if len(args.app_exclude) > 0:
            for excl in args.app_exclude:
                args.pair = [p for p in args.pair if excl not in p]

    elif args.how == 'dynamic':
        # expand all the multi-kernel benchmarks into individual kernels
        updated_pairs = []
        for pair in args.pair:
            apps = pair.split('+')

            def expand_bench(app):
                expanded = []
                if app in const.multi_kernel_app:
                    [expanded.append("{0}:{1}".format(app, kidx)) for kidx in
                     range(1, const.multi_kernel_app[app] + 1)]
                else:
                    expanded.append(app)
                return expanded

            list_1 = expand_bench(apps[0])
            list_2 = expand_bench(apps[1])
            cross_list = ["{0}+{1}".format(k1, k2) for k2 in list_2 for k1 in
                          list_1]
            updated_pairs += cross_list

        args.pair = updated_pairs


def main():
    parse_args()

    # handle all pairs and multi-kernel pairs in case of dynamic sharing
    process_pairs()

    # Keep track of total jobs launched
    job_count = 0
    pair_count = 0

    base_config = "TITANV-PAE-SEP_RW-CONCURRENT"

    if args.how == 'smk':
        len_configs = [process_smk(pair, base_config) for pair in args.pair]
    elif args.how == 'static':
        len_configs = [process_static(pair, base_config) for pair in args.pair]
    elif args.how == 'dynamic':
        len_configs = [process_dynamic(pair) for pair in args.pair]
    elif args.how == 'inter':
        len_configs = [process_inter(pair) for pair in args.pair]
    elif args.how == 'ctx':
        if args.num_slice <= 1:
            print('Context ratio option: num_slice is out of range. '
                  'Must be greater than 1.')
            sys.exit(2)

        # cap cycles are calculated from seq-multi.csv
        df_seq_multi = pd.read_csv(os.path.join(const.DATA_HOME,
                                                'csv/seq-multi.csv'))
        help_iso.process_config_column('kidx', df=df_seq_multi)
        df_seq_multi.set_index(['pair_str', 'kidx'], inplace=True)

        len_configs = \
            [process_ctx(pair, base_config, df_seq_multi) for pair in args.pair]
    else:
        print('Invalid sharing config how.')
        sys.exit(1)

    pair_count += sum(map(lambda x: x > 0, len_configs))
    job_count += sum(len_configs)

    print('\n')
    print('>' * 10, 'Summary', '<' * 10)
    print('Total app pairs considered:', pair_count)
    print('Total jobs attempted to launch:', job_count)


if __name__ == "__main__":
    main()
