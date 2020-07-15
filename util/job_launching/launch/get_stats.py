import re
import argparse
import os
import oyaml as yaml
import glob
import multiprocessing
import sys
from joblib import Parallel, delayed

import job_launching.launch.common as common
import data.scripts.common.constants as const

MAX_BREAK_TOKEN = r"GPGPU-Sim: \*\* break due to reaching " \
                  r"the maximum cycles \(or instructions\) \*\*"


def load_stats_yamls(filename):
    collections = yaml.load(open(filename), Loader=yaml.FullLoader)

    if 'scalar' in collections:
        for stat in collections['scalar']:
            collections['scalar'][stat] = \
                (re.compile(collections['scalar'][stat]), 'scalar')

    if 'vector' in collections:
        for stat in collections['vector']:
            collections['vector'][stat] = \
                (re.compile(collections['vector'][stat]), 'vector')

        collections['scalar'].update(collections['vector'])

    return collections['scalar']


# Print helper functions. #
def pretty_print(*str):
    print("get_stats.py >>>", *str)


def parse_args():
    parser = argparse.ArgumentParser("Grab stats from GPGPU-Sim output files "
                                     "and save them into csv.")

    parser.add_argument("--run_dir",
                        help="The directory where gpusim outputs exist. The "
                             "expected structured of this folder: "
                             "app/config/..",
                        default="", required=True)

    parser.add_argument("--format", choices=["stats", "timeline"],
                        default="stats",
                        help="Parsed file format. `stats` for csv files of "
                             "gpusim stats. `timeline` for timestamps of "
                             "kernel execution")

    parser.add_argument("--stats_yml", default="",
                        help="If format is `stats`, the yaml file that defines "
                             "the stats you want to collect")

    parser.add_argument("--exclude", default=[], nargs='+',
                        help="Exclude configs with these strings. "
                             "Accept space separated list.")

    parser.add_argument("--include", default=[], nargs='+',
                        help='Exclude configs with these strings. '
                             'Accept space separated list.')

    parser.add_argument("--output", default="result.csv",
                        help="If format is `stats`, "
                             "the logfile to save csv to.")

    parser.add_argument("--skip_hit_max", action="store_true",
                        help="Skip files that hit max cycles/instructions.")

    parser.add_argument("--debugging", action="store_true",
                        help="If debugging is turned on, failed logs will "
                             "have the last line printed to the console.")

    parser.add_argument("--latest", action="store_true",
                        help="Only parse the latest log in each folder.")

    args = parser.parse_args()
    return args


def process_yamls(args):
    # Deal with config, app and stats yamls
    common.load_defined_yamls()

    this_directory = os.path.dirname(os.path.realpath(__file__))

    if args.format == "stats":
        args.stats_yml = common.file_option_test(
            args.stats_yml,
            os.path.join(this_directory, "stats", "multi.yml"),
            this_directory)

        stats_to_pull = load_stats_yamls(args.stats_yml)

        return stats_to_pull
    else:
        return {}


# Parse the filename of gpusim output file to get
# app_and_args, config, gpusim_version and jobid
# pair_str is the name of the parent directory, it is 
# used to identify where app_and_args ends and config begins
def parse_outputfile_name(logfile):
    logfile_name = os.path.basename(logfile)

    # naming convention:
    # gpusim_log = job_name.job_id.log
    # job_name = app-config-commit_id
    dot_split = logfile_name.split(".")
    job_id = dot_split[1]

    job_name = dot_split[0]
    dash_split = job_name.split("-")

    gpusim_version = dash_split[-1]

    return gpusim_version, job_id


# Do a quick 10000-line reverse pass to
# make sure the simualtion thread finished
def has_exited(gpusim_logfile):
    exit_str = "GPGPU-Sim: \*\*\* exit detected \*\*\*"

    with open(gpusim_logfile, errors='ignore') as f:
        bytes_to_read = int(256*1024*1024)
        file_size = int(os.stat(gpusim_logfile).st_size)
        if file_size > bytes_to_read:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell()-bytes_to_read, os.SEEK_SET)

        lines = f.readlines()

        max_line_count = 300
        for line in reversed(lines):
            max_line_count -= 1
            if max_line_count < 0:
                break

            exit_match = re.match(exit_str, line)
            if exit_match:
                return True

    return False


def grep_timeline(log, config):
    hit_max = False
    result = []

    with open(log, "r") as f:
        for line in f:
            if re.search(MAX_BREAK_TOKEN, line):
                hit_max = True

            if re.search(r"ended @", line):
                result.append(line)

    return hit_max, (config, "".join(result))


def collect_stats(outputfile, stats_to_pull, app, config):
    hit_max = False
    stat_map = {}

    f = open(outputfile, errors='ignore')
    lines = f.readlines()

    # If found_uid is true and we encounter a second kernel_uid metric,
    # we should stop collecting stats from this file. This is to avoid parsing
    # two separate blocks of gpusim metrics. Normally this should not
    # happen unless the kernel pair finished almost at the same time. Print
    # out some warning in case this is an error.
    found_uid = False
    should_parse = True
    kernel_uid_token = r"kernel_launch_uid\s=\s"

    for line in lines:
        # If we ended simulation due to too many insn -
        # ignore the last kernel launch, as it is no complete.
        # Note: This only appies if we are doing kernel-by-kernel stats
        if re.match(MAX_BREAK_TOKEN, line):
            # print("get_stats.py >>> Found Max Insn reached in {0}."
            #       .format(outputfile))
            hit_max = True

        if re.match(kernel_uid_token, line):
            if found_uid:
                print("Warning: {} contains more than one gpusim output "
                      "block.".format(outputfile))
                should_parse = False
            else:
                found_uid = True

        if not should_parse:
            continue

        for stat_name, token in stats_to_pull.items():
            existance_test = token[0].search(line.rstrip())
            if existance_test:
                num_groups = len(existance_test.groups())

                number = existance_test.groups()[-1].strip()
                # avoid conflicts with csv commas
                number = number.replace(',', 'x')

                if token[1] == 'scalar':
                    stat_map[stat_name] = number
                else:
                    if num_groups == 1:
                        # Vector style: not associated with stream
                        # e.g. memory bandwidth usage per channel
                        if stat_name not in stat_map:
                            stat_map[stat_name] = []

                        stat_map[stat_name].append(number)
                    else:
                        # Vector style: associated with stream and kernel index
                        # e.g. runtime
                        if stat_name not in stat_map:
                            # Hard-code three different streams
                            stat_map[stat_name] = [[], [], []]

                        stream_id = int(existance_test.group(1))
                        # kidx in gpgpusim output starts from 0
                        kidx = int(existance_test.group(2))

                        if len(stat_map[stat_name][stream_id]) > kidx:
                            print('Invalid multi-dimension metric',
                                  stat_name, 'in', outputfile)
                            print('stream', stream_id)
                            print('kidx', kidx,
                                  'len', len(stat_map[stat_name][stream_id]))
                            sys.exit(1)
                        elif len(stat_map[stat_name][stream_id]) < kidx:
                            # GPGPU-Sim skipped printing stats, need to fill
                            # in some zeros
                            zeros = ['0'] * (kidx - 1 -
                                           len(stat_map[stat_name][stream_id]))
                            stat_map[stat_name][stream_id].extend(zeros)

                        stat_map[stat_name][stream_id].append(number)

    f.close()

    # get parameters from the file name and parent directory
    gpusim_version, job_Id = parse_outputfile_name(outputfile)

    # build a csv string to be written to file from this output
    csv_str = app + ',' + config + ',' + \
              gpusim_version + ',' + job_Id

    for stat in stats_to_pull:
        csv_str += ','
        if stat in stat_map:
            if stats_to_pull[stat][1] == 'scalar':
                csv_str += stat_map[stat]
            else:
                if isinstance(stat_map[stat][0], list):
                    # multi-dimensional array
                    csv_str += '['

                    for stream in stat_map[stat]:
                        csv_str += '[' + ' '.join(stream) + '] '

                    csv_str += ']'
                else:
                    csv_str += '[' + ' '.join(stat_map[stat]) + ']'
        else:
            if stats_to_pull[stat][1] == 'scalar':
                csv_str += '0'
            else:
                csv_str += '[0]'

    return hit_max, csv_str


def parse_app_files(app, args, stats_to_pull):
    result_list = []
    found_failed_app = False
    file_count = 0
    hit_max_count = 0

    stats_list = list(stats_to_pull.keys())
    app_path = os.path.join(args.run_dir, app)
    configs = [sd for sd in os.listdir(app_path) if
               os.path.isdir(os.path.join(app_path, sd))]

    # go into each config directory
    for config in configs:
        # check exclude strings
        if len(args.exclude) > 0 and \
                any(exclude in config for exclude in args.exclude):
            continue

        # check must include strings
        if len(args.include) > 0 and \
                not all(include in config for include in args.include):
            continue

        config_path = os.path.join(app_path, config)

        # for this config path we need to find the latest modified
        # output log file and parse it
        gpusim_logs = glob.glob(os.path.join(config_path, '*.log'))

        # sort logs based on time stamps
        def get_timestamp(x):
            return os.path.getctime(x)
        gpusim_logs = sorted(gpusim_logs, reverse=True, key=get_timestamp)

        if args.latest:
            gpusim_logs = gpusim_logs[:1]

        found_valid_log = False
        log = ''
        for latest_log in gpusim_logs:
            log = latest_log
            # do a reverse pass to check whether simulation exited
            if not has_exited(log):
                continue
            else:
                found_valid_log = True
                break

        if not found_valid_log:
            app_str = '+'.join(re.split(r'-(?=\D)', app))

            found_failed_app = True
            # Print the last line of log for debugging purposes
            if args.debugging:
                last_line = open(log).readlines()[-1]
                pretty_print("({0}, {1}): {2}".format(app_str, config,
                                                      last_line))
            else:
                pretty_print(
                    "--app {0} --config {1} log does not contain a "
                    "terminating string from GPGPU-Sim. Skip.".format
                    (app_str, config))

            # continue to the next config folder
            continue

        if args.format == "stats":
            hit_max, result_str = collect_stats(log, stats_to_pull, app, config)
        elif args.format == "timeline":
            hit_max, result_str = grep_timeline(log, config)
        else:
            print("Invalid format.")
            sys.exit(1)

        if hit_max:
            hit_max_count += 1

        if args.skip_hit_max and hit_max:
            # Do not print hit max results into csv
            continue

        result_list.append(result_str)
        file_count += 1

    return result_list, found_failed_app, file_count, hit_max_count


def main():
    # parse the arguments
    args = parse_args()

    if not os.path.isdir(args.run_dir):
        exit("Launch run dir " + args.run_dir + " does not exist.")
        exit(1)

    # process app, config and stat yamls
    stats_to_pull = process_yamls(args)

    apps = [d for d in os.listdir(args.run_dir) if
            os.path.isdir(os.path.join(args.run_dir, d))]
    # Exclude folder name 'gpgpu-sim-builds'
    if 'gpgpu-sim-builds' in apps:
        apps.remove('gpgpu-sim-builds')

    # book keeping numbers:
    file_count = 0
    hit_max_count = 0
    failed_apps = set()
    good_apps = set()
    max_apps = set()

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(parse_app_files)(app, args, stats_to_pull) for app in apps)

    csv_total = []
    for idx, app_result in enumerate(results):
        # app_result: string, failed, file_count, hit_max_count

        if app_result[2] > 0:
            if args.format == "stats":
                csv_total += app_result[0]
            elif args.format == "timeline":
                app_dir = os.path.join(const.DATA_HOME, 'timeline', apps[idx])
                if not os.path.exists(app_dir):
                    os.makedirs(app_dir)

                for config, timeline_str in app_result[0]:
                    timeline_path = os.path.join(app_dir,
                                                 '{}.txt'.format(config))
                    with open(timeline_path, "w") as f:
                        f.write(timeline_str)

        if app_result[1]:
            failed_apps.add(apps[idx])
        else:
            good_apps.add(apps[idx])

        if app_result[3] > 0:
            max_apps.add(apps[idx])

        file_count += app_result[2]
        hit_max_count += app_result[3]

    if args.format == "stats":
        f_csv = open(args.output, "w+")

        f_csv.write(
            'pair_str,config,gpusim_version,jobId,' + ','.join(
                stats_to_pull.keys()) + '\n')
        f_csv.write("\n".join(csv_total))

        f_csv.close()

        print(('-' * 100))
        pretty_print("Write to file {0}".format(args.output))

    print(('-' * 100))
    pretty_print("Successfully parsed {0} files".format(file_count))
    pretty_print("{0} files hit max cycle count.".format(hit_max_count))
    actual_failed = failed_apps - good_apps
    pretty_print("{0} failed simulation: {1}".format
                 (len(actual_failed), ','.join(list(actual_failed))))
    pretty_print("{0} apps hit max cycles: {1}".format(
        len(max_apps), ','.join(list(max_apps))))

    print(('-' * 100))


if __name__ == '__main__':
    main()
