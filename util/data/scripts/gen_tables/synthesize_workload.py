from random import seed
from random import choices
import os
import oyaml

import data.scripts.common.constants as const

REPEAT = 0
FRESH = 1


# Create 10 different multi-kernel synthetic workloads
def synthesize(rseed=1, num_apps=10, pRepeat=0.4, num_benchmarks=5):
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
                if choices([REPEAT, FRESH], weights=[pRepeat, 1 - pRepeat])[0] \
                        == REPEAT:
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
    synthesize()


if __name__ == '__main__':
    main()
