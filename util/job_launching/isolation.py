import argparse
import subprocess
import pandas as pd
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
    parser.add_argument('--mig', action='store_true',
                        help='Run mig experiments or not.')
    parser.add_argument('--mig_mem', action='store_true',
                        help='Run mig memory only experiments or not.')
    parser.add_argument('--bypassl2', action='store_true',
                        help='Do a bypass of l2 cache if l2 miss rate of the \
                        app is above the l2d_bypass_threshold.')

    parser.add_argument('--cta_configs', default=4, type=int,
                        help='Sweeping step of CTAs/SM for intra-SM sharing.')
    parser.add_argument('--sm_configs', default=4, type=int,
                        help='Sweeping step of SMs for inter-SM sharing.')
    parser.add_argument('--mig_configs', default=4, type=int,
                        help='Sweeping step of SM and mem channels for MIG '
                             'sharing.')

    parser.add_argument('--no_launch', default=False, action='store_true',
                        help='Do not actually trigger job launching.')

    parser.add_argument('--env', default='eco', choices=['eco', 'vector'],
                        help='Environment to launch.')

    results = parser.parse_args()

    return results


achieved_sm = const.num_sm_volta
mem_channels = const.num_mem_channels_volta

args = parse_args()

if args.apps[0] == 'all':
    all_apps = list(const.kernel_yaml.keys())

    last_index = min(len(all_apps), args.count + args.id_start)
    args.apps = all_apps[args.id_start:last_index]

# Load seq run results for find out whether the kernels should bypass L2D
df_seq = pd.read_pickle(os.path.join(const.DATA_HOME, 'pickles/seq.pkl'))
df_seq.set_index('pair_str', inplace=True, drop=True)

df_seq_multi = pd.read_pickle(os.path.join(const.DATA_HOME,
                                           'pickles/seq-multi.pkl'))
df_seq_multi.set_index(['pair_str', '1_kidx'], inplace=True, drop=True)

for app in args.apps:
    # Check if this is part of a multi-kernel app
    sp_kernel = app.split(':')

    if len(sp_kernel) > 1:
        if sp_kernel[0] not in const.multi_kernel_app:
            print("{} is not in multi-kernel map. Skip.", sp_kernel[0])
            continue
        else:
            kernels = [app]
    else:
        if app in const.multi_kernel_app:
            kernels = ["{0}:{1}".format(app, kidx)
                       for kidx in const.kernel_yaml[app].keys()]
        elif app in const.kernel_yaml.keys():
            kernels = [app]
        else:
            print("{0} is not in application map. Skip.".format(app))
            continue


    def launch_job(sm_config, jobname, kernel):
        configs = ["-".join([const.base_config, sm]) for sm in sm_config]

        split_kernel = kernel.split(':')
        if len(split_kernel) > 1:
            # This is a kernel part of a multi-kernel benchmark
            # append skip kidx config to do performance simulation only on
            # this kernel
            jobname += '-multi'
            bench = split_kernel[0]
            kidx = int(split_kernel[1])
            num_kernel = const.get_num_kernels(bench)
            configs = ["-".join([cfg,
                                 "MIX_0:{}:0_KIDX".format(kidx),
                                 "NUM_0:{}:0_KERNEL".format(num_kernel)]
                                )
                       for cfg in configs]

            if args.bypassl2:
                # Check if we should disable l2d
                configs = [cfg + "-BYPASS_L2D_S1" if
                           df_seq_multi.loc[(bench, kidx)]['l2_miss_rate'] >
                           const.l2d_bypass_threshold
                           else cfg for cfg in configs]
        else:
            if args.bypassl2:
                configs = [cfg + "-BYPASS_L2D_S1" if
                           df_seq.loc[kernel]['l2_miss_rate']
                           > const.l2d_bypass_threshold
                           else cfg for cfg in configs]

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



    if args.intra:
        for k in kernels:
            _abs_max = const.get_achieved_cta(k)
            _step = max(1, _abs_max // args.cta_configs)

            intra_sm = ["INTRA_0:{0}:0_CTA".format(i)
                        for i in range(_step, _abs_max, _step)]
            intra_sm.append("INTRA_0:{0}:0_CTA".format(_abs_max))

            launch_job(intra_sm, 'isolation-intra', k)

    if args.inter:
        for k in kernels:
            _abs_max = achieved_sm
            _step = max(1, _abs_max // args.sm_configs)

            inter_sm = ["INTER_0:{0}:0_SM".format(i)
                        for i in range(_step, _abs_max, _step)]
            inter_sm.append("INTER_0:{0}:0_SM".format(_abs_max))

            print(app, inter_sm)

            launch_job(inter_sm, 'isolation-inter', k)

    if args.mig:
        for k in kernels:

            sm_step = max(1, achieved_sm // args.mig_configs)
            mem_channel_step = max(1, mem_channels // args.mig_configs)
            
            sm_values = [i for i in range (sm_step, achieved_sm, sm_step)]
            sm_values.append(achieved_sm)

            mem_values = [i for i in range (mem_channel_step, mem_channels, 
                mem_channel_step)]
            mem_values.append(mem_channels)

            mig_params = ["MIG_{0}_MEM-MIG_{1}_SM".format(
                mem_values[i], sm_values[i]) for i in range(len(sm_values))]

            print(app, mig_params)

            launch_job(mig_params, 'mig_sweep', k)
    if args.mig_mem:
        for k in kernels:

            mem_channel_step = max(1, mem_channels // args.mig_configs)
            
            mem_values = [i for i in range (mem_channel_step, mem_channels, 
                mem_channel_step)]
            mem_values.append(mem_channels)

            mig_mem_params = ["MIG_{0}_MEM".format(
                mem_values[i]) for i in range(len(mem_values))]

            print(app, mig_mem_params)

            launch_job(mig_mem_params, 'mig_mem_sweep', k)
