import re
import argparse
import os
import job_launching.launch.common as common
import oyaml as yaml
import glob

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"


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

    parser.add_argument("--parent_run_dir",
                        help="The directory where run-* exist.",
                        default="", required=True)

    parser.add_argument("--launch_name",
                        help="Simulation name. Controls name of run folder",
                        default="", required=True)

    parser.add_argument("--stats_yml", default="",
                        help="The yaml file that defines the stats "
                             "you want to collect", required=True)

    parser.add_argument("--exclude", default="", nargs='+',
                        help="Exclude configs with these strings. "
                             "Comma separated list.")

    parser.add_argument("--include", default='', nargs='+',
                        help='Exclude configs with these strings. '
                             'Comma separated list.')

    parser.add_argument("--output", default="result.csv",
                        help="The logfile to save csv to.", required=True)

    args = parser.parse_args()
    return args


def process_yamls(args):
    # Deal with config, app and stats yamls
    common.load_defined_yamls()

    args.stats_yml = common.file_option_test(args.stats_yml,
                                             os.path.join(this_directory,
                                                          "stats",
                                                          "example_stats.yml"),
                                             this_directory)

    stats_to_pull = load_stats_yamls(args.stats_yml)

    # args.exclude = list(filter(None, args.exclude))
    # args.include = list(filter(None, args.include))

    return stats_to_pull


# Parse the filename of gpusim output file to get
# app_and_args, config, gpusim_version and jobid
# pair_str is the name of the parent directory, it is 
# used to identify where app_and_args ends and config begins
def parse_outputfile_name(logfile_name):
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

    with open(gpusim_logfile) as f:
        bytes_to_read = int(250 * 1024 * 1024)
        file_size = int(os.stat(gpusim_logfile).st_size)
        if file_size > bytes_to_read:
            f.seek(-bytes_to_read, os.SEEK_END)

        lines = f.readlines()

        max_line_count = 10000
        for line in reversed(lines):
            max_line_count -= 1
            if max_line_count < 0:
                break

            exit_match = re.match(exit_str, line)
            if exit_match:
                return True

    return False


def collect_stats(outputfile, stats_to_pull):
    stat_map = {}

    f = open(outputfile)
    lines = f.readlines()

    for line in lines:
        # If we ended simulation due to too many insn -
        # ignore the last kernel launch, as it is no complete.
        # Note: This only appies if we are doing kernel-by-kernel stats
        last_kernel_break = re.match(
            r"GPGPU-Sim: \*\* break due to "
            r"reaching the maximum cycles \(or instructions\) \*\*", line)
        if last_kernel_break:
            print("NOTE::::: Found Max Insn reached in {0}."
                  .format(outputfile))

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

    f.close()

    return stat_map


def main():
    # parse the arguments
    args = parse_args()

    # process stat yamls
    stats_to_pull = process_yamls(args)
    stats_list = list(stats_to_pull.keys())

    # directory containing all folders where we need the files from
    # get all app directory names, e.g. "cut_sgemm-1-cut_wmma-0"
    # each of these directory names also represents an app_and_args
    run_dir = common.get_run_dir(args.parent_run_dir,
                                 args.launch_name)

    if not os.path.isdir(run_dir):
        exit("Launch run dir " + run_dir + " does not exist.")
        exit(1)

    app_names = [d for d in os.listdir(run_dir) if
                 os.path.isdir(os.path.join(run_dir, d))]
    # Exclude folder name 'gpgpu-sim-builds'
    if 'gpgpu-sim-builds' in app_names:
        app_names.remove('gpgpu-sim-builds')

    f_csv = open(args.output, "w+")
    print(('-' * 100))
    f_csv.write(
        'pair_str,config,gpusim_version,jobId,' + ','.join(stats_list) + '\n')

    for app_name in app_names:
        app_path = os.path.join(run_dir, app_name)
        configs = [sd for sd in os.listdir(app_path) if
                   os.path.isdir(os.path.join(app_path, sd))]

        # go into each config directory
        for config in configs:
            config_path = os.path.join(app_path, config)

            # for this config path we need to find the latest modified
            # output log file and parse it
            log_file_names = glob.glob(os.path.join(config_path, '*.log'))
            last_mod_dates = [os.path.getmtime(x) for x in log_file_names]
            last_mod_index = last_mod_dates.index(max(last_mod_dates))
            last_log = log_file_names[last_mod_index]

            # do a reverse pass to check whether simulation exited
            if not has_exited(last_log):
                pretty_print("Detected that {0} does not contain a "
                             "terminating string from GPGPU-Sim. Skip.".format
                             (last_log))
                continue

            # get parameters from the file name and parent directory
            gpusim_version, job_Id = parse_outputfile_name(last_log)

            # build a csv string to be written to file from this output
            csv_str = app_name + ',' + config + ',' + \
                      gpusim_version + ',' + job_Id

            stat_map = collect_stats(last_log, stats_to_pull)

            for stat in stats_list:
                if stat in stat_map:
                    if stats_to_pull[stat][1] == 'scalar':
                        csv_str += ',' + stat_map[stat]
                    else:
                        csv_str += ',[' + ' '.join(stat_map[stat]) + ']'
                else:
                    csv_str += ',' + '0'

            # append information from this output file to the csv file
            f_csv.write(csv_str + '\n')

    f_csv.close()
    print(("Write to file {0}".format(args.output)))
    print(('-' * 100))

if __name__ == '__main__':
    main()
