import argparse
import os
import sys
import glob
import subprocess
import re
from job_launching.constant import *


def parse_args():
    parser = argparse.ArgumentParser("Rerun jobs with the same app and gpusim "
                                     "config if the job failed.")

    parser.add_argument('--run_dir', required=True,
                        help="The run dir to search for failed jobs. It's in "
                             "the format of run-<launch_name>.")

    parser.add_argument('--app_match', default='',
                        help='Only directories that include this app are '
                             'checked.')

    parser.add_argument('--bench_home', default=DEFAULT_BENCH_HOME,
                        help='Benchmark home folder.')

    parser.add_argument('--no_launch', default=False, action='store_true',
                        help='Do not actually trigger job launching.')

    parser.add_argument('--env', default='eco', choices=['eco', 'vector'],
                        help='Environment to launch.')

    args = parser.parse_args()
    return args


def get_subdir(directory):
    result = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(
        directory, d))]
    return result


def main():
    args = parse_args()

    if not os.path.isdir(args.run_dir):
        print("Invalid run_dir specified.")
        sys.exit(1)

    apps = get_subdir(args.run_dir)

    if 'gpgpu-sim-builds' in apps:
        apps.remove('gpgpu-sim-builds')

    for app in apps:
        if args.app_match != '' and args.app_match not in app:
            continue

        app_path = os.path.join(args.run_dir, app)
        configs = get_subdir(app_path)

        for config in configs:
            config_path = os.path.join(app_path, config)
            logs = glob.glob(os.path.join(config_path, '*.log'))

            job_ids = [os.path.basename(log).split('.')[1] for log in logs]

            # Query job state from slurm
            cmd = ['sacct', '-j', ','.join(job_ids), '--format=state',
                   '--parsable2']
            p = subprocess.run(cmd, stdout=subprocess.PIPE)

            output = p.stdout.decode("utf-8").splitlines()

            # Only rerun the job if all log files correspond to failed jobs
            found_failed = False
            all_failed = True
            for line in output:
                strip_line = line.strip()
                if strip_line == 'State':
                    continue

                if strip_line == 'FAILED':
                    found_failed = True
                else:
                    all_failed = False

            if found_failed and all_failed:
                split_app = re.split(r'-(?=\D)', app)
                run_cmd = ['python3',
                           os.path.join(RUN_HOME, 'run_simulations.py'),
                           '--app', '+'.join(split_app),
                           '--config', config,
                           '--bench_home', args.bench_home,
                           '--launch_name', re.sub('run-', '', args.run_dir),
                           '--env', args.env,
                           '--overwrite',
                           ]

                if args.no_launch:
                    run_cmd.append('--no_launch')

                p = subprocess.run(run_cmd, stdout=subprocess.PIPE)

                if not args.no_launch:
                    print(p.stdout.decode("utf-8"))


if __name__ == "__main__":
    main()
