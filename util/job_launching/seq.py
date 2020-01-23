import os
import subprocess
import argparse

from job_launching.constant import *
import data.scripts.common.constants as const


def parse_args():
    parser = argparse.ArgumentParser("Run app in isolation mode (vanilla)")

    parser.add_argument('--app', nargs='+', default=const.app_dict.keys(),
                        help='Apps to run')
    parser.add_argument('--bench_home', default=DEFAULT_BENCH_HOME,
                        help='Benchmark home folder.')
    parser.add_argument('--no_launch', default=False, action='store_true',
                        help='Do not actually trigger job launching.')
    parser.add_argument('--random', default=False, action='store_true',
                        help='Use random hashing for memory partition.')
    parser.add_argument('--env', default='eco', choices=['eco', 'vector'],
                        help='Launch environment. Either eco for torque '
                             'or vector for slurm.')

    results = parser.parse_args()

    return results


args = parse_args()

if args.random:
    config_str = "TITANV-SEP_RW-RANDOM"
    jobname = 'seq-rand'
else:
    config_str = "TITANV-SEP_RW"
    jobname = 'seq'

for benchmark in args.app:
    cmd = ['python3',
            os.path.join(RUN_HOME, 'run_simulations.py'),
            '--app', benchmark,
            '--config', config_str,
            '--bench_home', args.bench_home,
            '--launch_name', jobname,
            '--env', args.env
            ]
    if args.no_launch:
        cmd.append('--no_launch')

    print(cmd)
    p = subprocess.run(cmd, stdout=subprocess.PIPE)

    print(p.stdout.decode("utf-8"))
