import re
import argparse
import os
import job_launching.launch.common as common
import oyaml as yaml
import glob
import threading
import multiprocessing
from joblib import Parallel, delayed
from time import sleep


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

    parser.add_argument("--stats_yml", default="",
                        help="The yaml file that defines the stats "
                             "you want to collect", required=True)

    parser.add_argument("--exclude", default=[], nargs='+',
                        help="Exclude configs with these strings. "
                             "Accept space separated list.")

    parser.add_argument("--include", default=[], nargs='+',
                        help='Exclude configs with these strings. '
                             'Accept space separated list.')

    parser.add_argument("--output", default="result.csv",
                        help="The logfile to save csv to.", required=True)

    args = parser.parse_args()
    return args


def process_yamls(args):
    # Deal with config, app and stats yamls
    common.load_defined_yamls()

    this_directory = os.path.dirname(os.path.realpath(__file__))
    args.stats_yml = common.file_option_test(args.stats_yml,
                                             os.path.join(this_directory,
                                                          "stats",
                                                          "example_stats.yml"),
                                             this_directory)

    stats_to_pull = load_stats_yamls(args.stats_yml)

    return stats_to_pull


# Parse the filename of gpusim output file to get
# app_and_args, config, gpusim_version and jobid
# pair_str is the name of the parent directory, it is 
# used to identify where app_and_args ends and config begins
def parse_outputfile_name(logfile, app, config):
    logfile_name = os.path.basename(logfile)

    # naming convention:
    # gpusim_log = job_name.job_id.log
    # job_name = app-config-commit_id
    logfile_name = logfile_name.replace(app+'-'+config+'-', '')

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


def collect_stats(outputfile, stats_to_pull):
    hit_max = False
    stat_map = {}

    f = open(outputfile, errors='ignore')
    lines = f.readlines()

    for line in reversed(lines):
        # If we ended simulation due to too many insn -
        # ignore the last kernel launch, as it is no complete.
        # Note: This only appies if we are doing kernel-by-kernel stats
        last_kernel_break = re.match(
            r"GPGPU-Sim: \*\* break due to "
            r"reaching the maximum cycles \(or instructions\) \*\*", line)
        if last_kernel_break:
            # print("NOTE::::: Found Max Insn reached in {0}."
            #       .format(outputfile))
            hit_max = True

        for stat_name, token in stats_to_pull.items():
            existance_test = token[0].search(line.rstrip())
            if existance_test:
                number = existance_test.group(1).strip()
                # avoid conflicts with csv commas
                number = number.replace(',', 'x')

                if token[1] == 'scalar':
                    stat_map[stat_name] = number
                else:
                    if stat_name not in stat_map:
                        stat_map[stat_name] = [number]
                    else:
                        stat_map[stat_name].append(number)

        if len(stat_map) == len(stats_to_pull):
            # we collected all the info, exit!
            break

    f.close()

    return hit_max, stat_map


def parse_app_files(app, args, stats_to_pull):
    csv_list = []
    failed = False
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
        timestamps = [os.path.getmtime(x) for x in gpusim_logs]
        latest_index = timestamps.index(max(timestamps))
        latest_log = gpusim_logs[latest_index]

        # do a reverse pass to check whether simulation exited
        if not has_exited(latest_log):
            pretty_print("Detected that {0} does not contain a "
                         "terminating string from GPGPU-Sim. Skip.".format
                         (latest_log))
            failed = True
            continue

        # get parameters from the file name and parent directory
        gpusim_version, job_Id = parse_outputfile_name(latest_log, app,
                                                       config)
        # build a csv string to be written to file from this output
        csv_str = app + ',' + config + ',' + \
                  gpusim_version + ',' + job_Id

        hit_max, stat_map = collect_stats(latest_log, stats_to_pull)

        for stat in stats_list:
            if stat in stat_map:
                if stats_to_pull[stat][1] == 'scalar':
                    csv_str += ',' + stat_map[stat]
                else:
                    csv_str += ',[' + ' '.join(stat_map[stat]) + ']'
            else:
                if stats_to_pull[stat][1] == 'scalar':
                    csv_str += ',' + '0'
                else:
                    csv_str += ',[0]'

        csv_list.append(csv_str)

        file_count += 1

        if hit_max:
            hit_max_count += 1

    return '\n'.join(csv_list), failed, file_count, hit_max_count


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

    f_csv = open(args.output, "w+")
    f_csv.write(
        'pair_str,config,gpusim_version,jobId,' + ','.join(
            stats_to_pull.keys()) + '\n')

    # book keeping numbers:
    file_count = 0
    hit_max_count = 0
    failed_apps = set()
    good_apps = set()

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(parse_app_files)(app, args, stats_to_pull) for app in apps)

    for idx, app_result in enumerate(results):
        # app_result: csv, failed, file_count, hit_max_count
        if app_result[2] > 0:
            f_csv.write(app_result[0])
            f_csv.write('\n')

        if app_result[1]:
            failed_apps.add(apps[idx])
        else:
            good_apps.add(apps[idx])

        file_count += app_result[2]
        hit_max_count += app_result[3]

    f_csv.close()

    print(('-' * 100))
    pretty_print("Write to file {0}".format(args.output))
    pretty_print("Successfully parsed {0} files".format(file_count))
    pretty_print("{0} files hit max cycle count.".format(hit_max_count))
    actual_failed = failed_apps - good_apps
    pretty_print("{0} failed simulation: {1}".format
                 (len(actual_failed), ','.join(list(actual_failed))))

    print(('-' * 100))


if __name__ == '__main__':
    main()
