import os
import subprocess
import argparse

from launch.constant import *


def parse_args():
    parser = argparse.ArgumentParser("Run app in isolation mode (vanilla)")

    parser.add_argument('--app', nargs='+', default=app_dict.keys(), help='Apps to run')
    parser.add_argument('--bench_home', default=DEFAULT_BENCH_HOME,
                        help='Benchmark home folder.')
    parser.add_argument('--no_launch', default=False, action='store_true',
                        help='Do not actually trigger job launching.')
    parser.add_argument('--random', default=False, action='store_true',
			help='Use random hashing for memory partition.')

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
    p = subprocess.run(['python3',
                        os.path.join(RUN_HOME, 'run_simulations.py'),
                        '-B', benchmark,
                        '-C', config_str,
                        '-E', args.bench_home,
                        '-N', jobname,
                        '-n' if args.no_launch else ''
                        ],
                       stdout=subprocess.PIPE)

    print(p.stdout.decode("utf-8"))


