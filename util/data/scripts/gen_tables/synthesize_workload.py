from random import seed
from random import choices
import os
import oyaml
import argparse

import data.scripts.common.constants as const

REPEAT = 0
FRESH = 1


def parse_args():
    parser = argparse.ArgumentParser('Synthesize workloads from '
                                     'micro-benchmarks.')
    parser.add_argument('--repeat',
                        action="store_true",
                        help='Make kernels repeat.')

    results = parser.parse_args()
    return results


# Create 10 different multi-kernel synthetic workloads
def synthesize(rseed=1, num_apps=10, pRepeat=0.4, num_benchmarks=5,
               repeat=False):
    seed(rseed)
    # Candidates are non multi-kernel benchmarks
    benchmarks = [benchmark for benchmark in const.kernel_yaml if benchmark
                  not in const.multi_kernel_app]

    apps = {}

    for app_idx in range(num_apps):
        app = []

        for bench_idx in range(num_benchmarks):
            if bench_idx == 0:
                # The first one cannot be repeat
                app += choices(benchmarks)
            else:
                # Figure out if we are repeating or selecting a fresh benchmark
                repeat_or_fresh = choices([REPEAT, FRESH],
                                          weights=[pRepeat, 1 - pRepeat])[0]
                if repeat and repeat_or_fresh == REPEAT:
                    app.append('repeat')
                else:
                    # Make sure we don't pick the same benchmark again
                    leftover = [benchmark for benchmark in benchmarks
                                if benchmark not in app]
                    app += choices(leftover)

        print(app)
        apps['syn-{}'.format(app_idx)] = app

    # write to yml file
    outfile = os.path.join(const.DATA_HOME,
                           'scripts/common/synthetic.yml')

    with open(outfile, 'w+') as f:
        f.write(oyaml.dump(apps, default_flow_style=False))

    return apps


def main():
    args = parse_args()
    synthesize(repeat=args.repeat)


if __name__ == '__main__':
    main()
