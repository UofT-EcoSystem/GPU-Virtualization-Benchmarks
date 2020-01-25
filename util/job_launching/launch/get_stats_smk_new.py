#!/usr/bin/env python

from optparse import OptionParser
import re
import os
import job_launching.launch.common as common
import math
import oyaml as yaml
import time

#millnames = ['', ' K', ' M', ' B', ' T']
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

#def millify(n):
#    n = float(n)
#    if math.isnan(n):
#        return "NaN"
#    if math.isinf(n):
#        return "inf"
#    millidx = max(0, min(len(millnames) - 1,
#                         int(math.floor(0 if n == 0
#                                        else math.log10(abs(n)) / 3))))
#    return '{:.3f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


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


# *********************************************************--
# main script start
# *********************************************************--


# Print helper functions. #
def pretty_print(*str):
    print("run_simulations.py >>>", *str)


def parse_args():
    parser = argparse.ArgumentParser("Invoke simulation jobs using job "
                                     "scheduler (Torque/Slurm).")

    parser.add_argument("--parent_run_dir", dest="parent_run_dir",
                      help="The directory where logfiles and run-* exist.",
                      default="", required=True)

    parser.add_argument("--launch_name", dest="launch_name",
                      help="Simulation name. Controls name of run folder",
                      default="", required=True)
    parser.add_argument("--stats_yml", dest="stats_yml", default="",
                      help="The yaml file that defines the stats "
                           "you want to collect", required=True)

    parser.add_argument("--log_format", dest="format", default="eco",
                      choices=['eco', 'vector'],
                      help="Logfile format: eco for torque style and vector for "
                           "slurm style.", required=True)

    parser.add_argument("--exclude-conf", dest="exclude", default="",
                      help="Exclude configs with these strings. "
                           "Comma separated list.")
    parser.add_argument("--include-conf", dest='include', default='',
                      help='Exclude configs with these strings. '
                           'Comma separated list.')

    parser.add_argument("--output", dest="outfile", default="result.csv",
                      help="The logfile to save csv to.", required=True)

    args = parser.parse_args()
    return args


# check if parent run dir exists under benchmark root
def sanity_check(args):
    if not os.path.exists(args.parent_run_dir):
        print('Benchmark root {0} does not contain a run/ folder. Exiting'
              .format(args.benchmark_root))
        exit(1)

    args.run_dir = common.get_run_dir(args.parent_run_dir,
                                         args.launch_name)
    args.log_dir = common.get_log_dir(args.parent_run_dir)

    if not os.path.isdir(args.run_dir):
        exit(args.run_dir +
             " does not exist - specify the run directory "
             "where the benchmark/config dirs exist")


def process_yamls(args):
    # 2. Deal with config, app and stats yamls
    common.load_defined_yamls()

    args.stats_yml = common.file_option_test(args.stats_yml,
                                                os.path.join(this_directory,
                                                             "stats",
                                                             "example_stats.yml"),
                                                this_directory)

    stats_to_pull = load_stats_yamls(args.stats_yml)

    args.exclude = list(filter(None, args.exclude.split(',')))
    args.include = list(filter(None, args.include.split(',')))

    return stats_to_pull




# 5. Parse and analyze job output
#stat_map = {}  # map: app+config+stat -> stat_value
# example of job name: 
#                               cut_sgemm-0         -             TITANV-SEP_RW   -       commit_6065f_modified_0                             .90.eco14-docker    .log
#                               app_ans_args                        config                      gpusim_version                                .jobid              .log
# example of log file:
#                               22:54:42    4.docker.gpusim       cut_sgemm-1            TITANV-SEP_RW-RANDOM                                                  commit_5282f_modified_0
#                               jobtime          jobId                 pair_str                config                                                             gpusim_version

#                               00:19:01    31.docker.gpusim      cut_wmma-1             TITANV-SEP_RW-CONCURRENT-INTRA_0:2:0_CTA-PARTITION_L2_0:0.5:0.5       commit_6065f_modified_0


# Parse the filename of gpusim output file to get 
# app_and_args, config, gpusim_version and jobid
# pair_str is the name of the parent directory, it is 
# used to identify where app_and_args ends and config begins
def parse_outputfile_name(outputfile):
    params = []
    full_split = outputfile.split(".")
    job_id = full_split[1] + full_split[2]
    name = full_split[0]
    split_name = name.split("-")
    #app_and_args = ""
    #config_index = 0
    #while app_and_args != pair_str:
    #    app_and_args += params[config_index]
    #    config_index += 1
    ## config_index is where config begins
    gpusim_version = split_name[-1]
    #gpusim_version_index = len(params) - 1
    ## config is everything between gpusim_version and end of app_and_args
    #config = "".join(params[config_index:gpusim_version_index])
    return (gpusim_version, job_id)


# Do a quick 10000-line reverse pass to
# make sure the simualtion thread finished
def reverse_pass(stat_map, pair_str, config_name, outputfile, outputfile_name, stats_to_pull):

    if not os.path.isfile(outputfile):
        print("WARNING - " + outputfile + " does not exist")

    # get parameters from the file name and parent directory
    gpusim_version, job_Id = parse_output_name(outputfile_name)
    app_and_args, config = pair_str, config_name

    SIM_EXIT_STRING = \
        "GPGPU-Sim: \*\*\* exit detected \*\*\*"

    exit_success = False
    MAX_LINES = 10000
    BYTES_TO_READ = int(250 * 1024 * 1024)
    count = 0

    f = open(outputfile)
    fsize = int(os.stat(outputfile).st_size)
    if fsize > BYTES_TO_READ:
        f.seek(-BYTES_TO_READ, os.SEEK_END)
    lines = f.readlines()

    for line in reversed(lines):
        count += 1
        if count >= MAX_LINES:
            break

        exit_match = re.match(SIM_EXIT_STRING, line)
        if exit_match:
            exit_success = True
            break
    del lines
    f.close()

    if not exit_success:
        print("Detected that {0} does not contain a terminating "
              "string from GPGPU-Sim. Skip.".format(outputfile))

    stat_map[app_and_args + config + 'gpusim_version'] = gpusim_version
    stat_map[app_and_args + config + 'jobid'] = jobId

    stat_found = set()
    files_parsed += 1
    bytes_parsed += os.stat(outputfile).st_size

    f = open(outputfile)
    lines = f.readlines()

    # print "Parsing File {0}. Size: {1}".format(outfile,
    # millify(os.stat(outfile).st_size))
    # reverse pass cuz the stats are at the bottom
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
                stat_found.add(stat_name)
                number = existance_test.group(1).strip()
                # avoid conflicts with csv commas
                number = number.replace(',', 'x')

                if token[1] == 'scalar':
                    stat_map[app_and_args + config + stat_name] = number
                else:
                    if app_and_args + config + stat_name not in stat_map:
                        stat_map[app_and_args + config + stat_name] = \
                            [number]
                    else:
                        stat_map[app_and_args + config + stat_name] \
                            .append(number)

    f.close()


def main():

    # parse the arguments
    args = parse_args()
    args.launch_name = args.launch_name.strip()

    # perform sanity check
    sanity_check(args)

    # process yamls 
    stats_to_pull = process_yamls(args)
    stats_list = list(stats_to_pull.keys())

    stat_map = {}
    f = open(args.outfile, "a+")
    print(('-' * 100))
    f.write('pair_str,config,gpusim_version,jobId,' + ','.join(stats_list) + '\n')


    # directory containing all folders where we need the files from
    run_dir = os.path.join(parent_dir, launch_name)
    # get all app directory names, e.g. "cut_sgemm-1-cut_wmma-0"
    # each of these directory names also represents an app_and_args
    app_names = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    for app_name in app_names:
        app_path = os.path.join(run_dir, app_name)
        configs = [sd for sd in os.listdir(app_path) if os.path.isdir(os.path.join(app_path, sd))]
        # go into each config directory
        for config in configs:
            config_path = os.path.join(app_path, config)
            # for this config path we need to find the latest modified output log file and parse it
            log_file_names = glob.glob(os.path.join(config_path, '*.log'))
            last_mod_dates = [os.path.getmtime(x) for x in log_file_names] 
            last_mod_index = last_mod_dates.index(max(last_mod_dates))
            last_log_name = log_file_names[last_mod_index]
            last_log = os.path.join(config_path, last_log_name)

            # do a reverse pass and fill out stat_map for this output
            reverse_pass(stat_map, app_name, config, last_log, last_log_name, stats_to_pull)
            # build a csv string to be written to file from this output
            csv_str = app_name + ',' \
                      + config + ',' \
                      + stat_map[app_name + config + 'gpusim_version'] + ',' \
                      + stat_map[app_name + config + 'jobid']

            for stat in stats_list:
                if app_name + config + stat in stat_map:
                    if stats_to_pull[stat][1] == 'scalar':
                        csv_str += ',' + stat_map[app_name + config + stat]
                    else:
                        csv_str += ',[' + ' '.join(
                            stat_map[app_name + config + stat]) + ']'
                else:
                    csv_str += ',' + '0'

            # append information from this output file to the csv file
            f.write(csv_str + '\n')


    print(("Write to file {0}".format(args.outfile)))
    print(('-' * 100))



    
# print out the csv file
#print(('-' * 100))
# just to make sure we print the stats in deterministic order,
# store keys of the map into a stats_list
#
#with open(options.outfile, 'w+') as f:
#    f.write('pair_str,config,gpusim_version,jobId,' + ','.join(stats_list)
#            + '\n')
#    print(('pair_str,config,gpusim_version,jobId,' + ','.join(stats_list)))
#
#    for app_str in apps_and_args:
#        for config in configs:
#            if app_str + config + 'gpusim_version' not in stat_map:
#                continue
#
#            csv_str = app_str + ',' \
#                      + config + ',' \
#                      + stat_map[app_str + config + 'gpusim_version'] + ',' \
#                      + stat_map[app_str + config + 'jobid']
#
#            for stat in stats_list:
#                if app_str + config + stat in stat_map:
#                    if stats_to_pull[stat][1] == 'scalar':
#                        csv_str += ',' + stat_map[app_str + config + stat]
#                    else:
#                        csv_str += ',[' + ' '.join(
#                            stat_map[app_str + config + stat]) + ']'
#                else:
#                    csv_str += ',' + '0'
#
#            f.write(csv_str + '\n')
#            print(csv_str)
#
#print(("Write to file {0}".format(args.outfile)))
#print(('-' * 100))

#print("Script exec time {0:.2f} seconds. {1} files and {2}B parsed. {3}B/s". \
#      format(duration, files_parsed, millify(bytes_parsed),
#             millify(float(bytes_parsed) / float(duration))))
