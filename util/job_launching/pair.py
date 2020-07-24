import argparse
import subprocess
import pandas as pd
import numpy as np
import sys
import oyaml as yaml
from job_launching.constant import *
import data.scripts.common.constants as const
import data.scripts.common.help_iso as help_iso
import data.scripts.gen_tables.gen_pair_configs as dynamic
import data.scripts.gen_tables.gen_inter_configs as inter

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
                        help="Apps to run. Accept 1) benchmark name, "
                             "2) `single` for all single-kernel benchmarks, "
                             "3) `multi` for all multi-kernel benchmarks"
                             "or 4) `syn` for synthetic workloads")
    parser.add_argument('--id_start', type=int, default=0,
                        help='For all pairs only. Starting pair id.')
    parser.add_argument('--count', type=int, default=20,
                        help='Max number of simulations to launch.')

    parser.add_argument('--app_match', default=[], nargs='+',
                        help='Select all pairs that include this app. Only '
                             'checked when all is passed to --pair.')
    parser.add_argument('--app_exclude', default=[], nargs='+',
                        help='Select all pairs that do not include this app. '
                             'Only checked when all is passed to --pair.')

    parser.add_argument('--how', choices=['smk', 'static', 'dynamic',
                                          'inter', 'ctx', 'lut', 'custream'],
                        help='How to partition resources between benchmarks.')
    parser.add_argument('--serial', action='store_true',
                        help='If how is lut, only run pairs where '
                             'kernel serialization is needed.')
    parser.add_argument('--num_slice', default=4, type=int,
                        help='If how is ctx, num_slice specifies how many '
                             'slice configs to sweep.')
    parser.add_argument('--intra_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/intra.pkl'),
                        help='If how is dynamic/lut, path to the intra pickle '
                             'file.')
    parser.add_argument('--dynamic_pkl',
                        default=os.path.join(const.DATA_HOME,
                                             'pickles/pair_dynamic.pkl'),
                        help='If how is lut/dynamic, path to the pair dynamic '
                             'pickle '
                             'file')
    parser.add_argument('--top',
                        action='store_true',
                        help='If how is dynamic, only select top candidates.')
    parser.add_argument('--qos',
                        type=float,
                        default=0.5,
                        help='If how is dynamic/inter, specify target qos.')
    parser.add_argument('--check_existing', action='store_true',
                        help='If how is dynamic/inter, only launch pairs that '
                             'do not exist in df_dynamic/df_inter.')
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


def process_custream(pair, config):
    configs = [config + '-CUDA_0:{}:0_STREAM'.format(const.LAUNCH_LATENCY),
               config + '-CUDA_0:0:{}_STREAM'.format(const.LAUNCH_LATENCY)]

    [launch_job(cfg, pair=pair) for cfg in configs]

    return 2


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
    pair_config_args += ['--qos', str(args.qos)]

    if args.top:
        pair_config_args += ['--top']

    # dynamic.main returns an array of candidate configs
    configs = dynamic.main(pair_config_args)

    split_kernels = [k.split(':') for k in apps]
    kidx = [sk[1] if len(sk) > 1 else 1 for sk in split_kernels]
    num_kernel = [const.get_num_kernels(sk[0]) for sk in split_kernels]

    configs = ["-".join([cfg,
                         "MIX_0:{0}:{1}_KIDX".format(kidx[0], kidx[1]),
                         "NUM_0:{0}:{1}_KERNEL".format(num_kernel[0],
                                                       num_kernel[1])]
                        )
               for cfg in configs]

    # Each app in pair should indicate which kernel is run for multi-kernel
    # benchmark
    pair = '+'.join([split_kernels[0][0], split_kernels[1][0]])

    multi = any(len(sk) > 1 for sk in split_kernels)

    if args.check_existing:
        df_dynamic = pd.read_pickle(args.dynamic_pkl)

        def found_in_pair_dynamic(_config):
            intra_1 = help_iso.check_config('1_intra', _config, default=0)
            intra_2 = help_iso.check_config('2_intra', _config, default=0)
            df_found = df_dynamic[(df_dynamic['pair_str'] == '-'.join(apps)) &
                                  (df_dynamic['1_intra'] == intra_1) &
                                  (df_dynamic['2_intra'] == intra_2)
                                  ]
            if df_found.empty:
                return False
            else:
                return True

        # Filter out existing configs
        configs = [c for c in configs if not found_in_pair_dynamic(c)]

    launch_job(*configs, pair=pair, multi=multi)

    return len(configs)


def get_cap_config(apps):
    max_cycles = 0

    for app in apps:
        sum_cycles = sum(const.get_seq_cycles(app))

        if sum_cycles > max_cycles:
            max_cycles = sum_cycles

    cycles = args.cap * max_cycles
    cap_config = 'CAP_{0}_CYCLE'.format(int(cycles))

    return cap_config


# LUT selects resource configs based on the best pair dynamic results
def process_lut(pair, base_config):
    import data.scripts.gen_tables.gen_lut_configs as lut_configs

    df_pair_dynamic = pd.read_pickle(args.dynamic_pkl)
    df_intra = pd.read_pickle(args.intra_pkl)

    apps = pair.split('+')

    # Build the CTA lookup table
    lut_matrix = lut_configs.get_lut_matrix(apps, df_pair_dynamic,
                                            df_intra)

    cta_configs = lut_matrix[0]
    serial = lut_matrix[2]
    lut = []

    if (args.serial and 1 in serial) or \
            (not args.serial and np.all(serial == 0)):
        # Columns are kernels for apps[0] and rows are kernels for apps[1]
        for row_idx, col_idx in np.ndindex(cta_configs[0].shape):
            entry = "0:{}:{}=0:{}:{}".format(col_idx + 1, row_idx + 1,
                                             cta_configs[0][row_idx, col_idx],
                                             cta_configs[1][row_idx, col_idx])
            lut.append(entry)

        lut_config = ",".join(lut)
        lut_config = "INTRA_{}_LUT".format(lut_config)

        # Number of kernels config
        num_kernel = [const.get_num_kernels(app) for app in apps]
        num_kernel_config = "NUM_0:{0}:{1}_KERNEL".format(num_kernel[0],
                                                          num_kernel[1])

        config = base_config + "-" + lut_config + "-" + num_kernel_config

        # Calculate max cycle bound
        cap_config = get_cap_config(apps)
        config += '-' + cap_config

        launch_job(config, pair=pair)

        return 1
    else:
        return 0


def process_inter(pair):
    apps = pair.split('+')
    # inter-SM sharing
    pair_config_args = ['--apps'] + apps
    pair_config_args.append('--print')
    pair_config_args += ['--cap', str(args.cap)]
    pair_config_args += ['--top']
    pair_config_args += ['--qos', str(args.qos)]

    configs = inter.main(pair_config_args)

    if args.check_existing:
        df_inter = const.get_pickle('pair_inter.pkl')

        def found_in_pair_inter(_config):
            inter_1 = help_iso.check_config('1_inter', _config, default=0)
            inter_2 = help_iso.check_config('2_inter', _config, default=0)
            df_found = df_inter[(df_inter['pair_str'] == '-'.join(apps)) &
                                (df_inter['1_inter'] == inter_1) &
                                (df_inter['2_inter'] == inter_2)
                                ]
            if df_found.empty:
                return False
            else:
                return True

        # Filter out existing configs
        configs = [c for c in configs if not found_in_pair_inter(c)]

    launch_job(*configs, pair=pair)

    return len(configs)


def find_ctx_configs(apps, num_slice):
    configs = []

    # CTX config
    # Get max resource usage from each bench in each app
    max_rsrc_usage = [const.get_dominant_usage(app) for app in apps]

    # 1. Make sure we can at least launch one TB of each kernel in each app
    max_usage_app = [max(app_usage, key=lambda x: x[1]) for app_usage in
                     max_rsrc_usage]
    # ctx does not care about which resource it is
    max_usage_app = [usage_tuple[1] for usage_tuple in max_usage_app]

    if sum(max_usage_app) > 1.0:
        print("No feasible ctx config for {}".format(apps))
        return []

    # Dice the remaining usage in multiple slices and check whether different
    # slice size leads to different cta quota for kernels
    remaining_usage = 1 - sum(max_usage_app)
    step = remaining_usage / num_slice
    previous_cta_setting = []

    for r in np.arange(max_usage_app[0], 1 - max_usage_app[1] + step, step):
        cta_setting = [const.get_cta_from_ctx(max_rsrc_usage[0], r, apps[0]),
                       const.get_cta_from_ctx(max_rsrc_usage[1], 1 - r,
                                              apps[1])]
        if cta_setting != previous_cta_setting:
            # Only launch another config if the cta setting differs
            previous_cta_setting = cta_setting
            configs.append('INTRA_0:{0}:{1}_RATIO'.format(r, 1 - r))

    return configs


def process_ctx(pair, base_config):
    apps = pair.split('+')

    # Get ctx possible configs
    configs = find_ctx_configs(apps, base_config, args.num_slice)

    if len(configs) == 0:
        return 0

    # Add base config
    configs = [base_config + '-' + c for c in configs]

    # Number of kernels config
    num_kernel = [const.get_num_kernels(app) for app in apps]
    num_kernel_config = "NUM_0:{0}:{1}_KERNEL".format(num_kernel[0],
                                                      num_kernel[1])

    configs = [c + '-' + num_kernel_config for c in configs]

    # Cycle cap config
    cap_config = get_cap_config(apps)
    configs = [c + '-' + cap_config for c in configs]

    launch_job(*configs, pair=pair)
    return len(configs)


def process_pairs():
    print("processing pairs")
    global args

    def pair_up(candidates):
        pairs = []
        for bench0 in candidates:
            for bench1 in candidates:
                if bench0 < bench1 and (bench0 in args.app_match or bench1 in
                        args.app_match):
                    # Make sure we don't pair up the same benchmarks with
                    # different inputs, except for synthetic workloads
                    # bench0_name = bench0.split('-')[0]
                    # bench1_name = bench1.split('-')[0]
                    # if bench0_name != bench1_name or bench0_name == 'syn':
                    pairs.append('+'.join([bench0, bench1]))

        print(pairs)
        if not args.id_start < len(pairs):
            print('Length of all pairs is {0} but id_start is {1}'
                  .format(len(pairs), args.id_start))
            exit(1)
        id_end = args.id_start + args.count

        if id_end > len(pairs):
            id_end = len(pairs)

        args.pair = pairs[args.id_start:id_end]

        # Remove all pairs in app_exclude list
        if len(args.app_exclude) > 0:
            for excl in args.app_exclude:
                args.pair = [p for p in args.pair if excl not in p]

    # Determine what app pairs to launch
    if args.pair[0] == 'single':
        benchmarks = [b for b in const.kernel_yaml.keys() if b not in
                      const.multi_kernel_app]
        pair_up(benchmarks)
    elif args.pair[0] == 'multi':
        benchmarks = const.multi_kernel_app.keys()
        pair_up(benchmarks)
    elif args.pair[0] == 'syn':
        syn_yml = yaml.load(
            open(os.path.join(THIS_DIR,
                              '../data/scripts/common/synthetic.yml')),
            Loader=yaml.FullLoader
        )
        benchmarks = syn_yml.keys()
        pair_up(benchmarks)

    if args.how == 'dynamic':
        # expand all the multi-kernel benchmarks into individual kernels
        updated_pairs = []
        for pair in args.pair:
            apps = pair.split('+')

            def expand_bench(app):
                expanded = []
                if app in const.multi_kernel_app:
                    [expanded.append("{0}:{1}".format(app, kidx)) for kidx
                     in const.kernel_yaml[app].keys()]
                else:
                    expanded.append(app)
                return expanded

            list_1 = expand_bench(apps[0])
            list_2 = expand_bench(apps[1])
            cross_list = ["{0}+{1}".format(k1, k2) for k2 in list_2 for k1
                          in list_1]
            updated_pairs += cross_list

        args.pair = updated_pairs


def main():
    parse_args()

    # handle all pairs and multi-kernel pairs in case of dynamic sharing
    process_pairs()

    # Keep track of total jobs launched
    job_count = 0
    pair_count = 0

    base_config = const.pair_base_config

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

        len_configs = [process_ctx(pair, base_config) for pair in args.pair]
    elif args.how == 'lut':
        len_configs = [process_lut(pair, base_config) for pair in args.pair]
    elif args.how == 'custream':
        len_configs = [process_custream(pair, base_config) for pair in
                       args.pair]
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
