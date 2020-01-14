#!/usr/bin/env python

from optparse import OptionParser
import re
import os
import common
import math
import oyaml as yaml
import time

millnames = ['', ' K', ' M', ' B', ' T']


def millify(n):
    n = float(n)
    if math.isnan(n):
        return "NaN"
    if math.isinf(n):
        return "inf"
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    return '{:.3f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


def load_stats_yamls(filename):
    collections = yaml.load(open(filename), Loader=yaml.FullLoader)

    if 'scalar' in collections:
        for stat in collections['scalar']:
            collections['scalar'][stat] = (re.compile(collections['scalar'][stat]), 'scalar')

    if 'vector' in collections:
        for stat in collections['vector']:
            collections['vector'][stat] = (re.compile(collections['vector'][stat]), 'vector')

        collections['scalar'].update(collections['vector'])

    return collections['scalar']


this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# *********************************************************--
# main script start
# *********************************************************--
start_time = time.time()
files_parsed = 0
bytes_parsed = 0

parser = OptionParser()

parser.add_option("-R", "--parent_run_dir", dest="parent_run_dir",
                  help="The directory where logfiles and run-* exist.", default="")
parser.add_option("-N", "--launch_name", dest="launch_name",
                  help="If you are launching run_simulations.py with the \"-N\" option" + \
                       " then you can run ./job_status.py with \"-N\" and it will" + \
                       " give you the status of the latest run with that name." + \
                       " if you want older runs from this name, then just point it directly at the" + \
                       " logfile with \"-l\"", default="")
parser.add_option("-S", "--stats_yml", dest="stats_yml", default="",
                  help="The yaml file that defines the stats you want to collect." + \
                       " by default it uses stats/example_stats.yml")

parser.add_option("-e", '--exclude-conf', dest='exclude', default='',
                  help='Exclude configs with these strings. Comma separated list.')
parser.add_option("-i", '--include-conf', dest='include', default='',
                  help='Exclude configs with these strings. Comma separated list.')


parser.add_option("-o", "--output", dest="outfile", default="result.csv",
                  help="The logfile to save csv to.")

# 1. Cmd parsing and sanity check
(options, args) = parser.parse_args()

options.launch_name = options.launch_name.strip()

# check if parent run dir exists under benchmark root
if not os.path.exists(options.parent_run_dir):
    print('Benchmark root {0} does not contain a run/ folder. Exiting'.format(options.benchmark_root))
    exit(1)

options.run_dir = common.get_run_dir(options.parent_run_dir, options.launch_name)
options.log_dir = common.get_log_dir(options.parent_run_dir)

if not os.path.isdir(options.run_dir):
    exit(options.run_dir + " does not exist - specify the run directory where the benchmark/config dirs exist")

# 2. Deal with config, app and stats yamls
common.load_defined_yamls()

options.stats_yml = common.file_option_test(options.stats_yml,
                                            os.path.join(this_directory, "stats", "example_stats.yml"),
                                            this_directory)

stats_to_pull = load_stats_yamls(options.stats_yml)

options.exclude = list(filter(None, options.exclude.split(',')))
options.include = list(filter(None, options.include.split(',')))

# 3. Look for matching log files

logfiles_directory = options.log_dir

if not os.path.exists(logfiles_directory):
    exit("Default logfile directory cannot be found")

all_logfiles = [os.path.join(logfiles_directory, f)
                for f in os.listdir(logfiles_directory) if (re.match(r'sim_log.*', f))]

if len(all_logfiles) == 0:
    exit("ERROR - No Logfiles in " + logfiles_directory)

named_sim = []
for logf in all_logfiles:
    match_str = r".*\/sim_log\..*\.{0}\..*".format(options.launch_name)
    if re.match(match_str, logf):
        named_sim.append(logf)

if len(named_sim) == 0:
    exit("Could not find logfiles for job with the name \"{0}\"".format(options.launch_name))

# Sort sim log files based on modification time so that new app + config pair will override the old one
named_sim.sort(key=os.path.getmtime)

print("Using logfiles " + str(named_sim))

# 4. Parse log files
configs = set()
apps_and_args = set()
# exes_and_args = set()
specific_jobIds = {}
stats = {}

for logfile in named_sim:
    if not os.path.isfile(logfile):
        exit("Cannot open Logfile " + logfile)

    with open(logfile) as f:
        for line in f:
            jobtime, jobId, pair_str, config, gpusim_version = line.split()

            # check exclude strings
            if options.exclude and any(exclude in config for exclude in options.exclude):
                continue

            if options.include and not all(include in config for include in options.include):
                continue

            configs.add(config)
            apps_and_args.add(pair_str)

            specific_jobIds[pair_str + config] = (jobId, gpusim_version)

# 5. Parse job outputs
stat_map = {}  # map: app+config+stat -> stat_value
for idx, app_and_args in enumerate(apps_and_args):
    for config in configs:
        if app_and_args + config not in specific_jobIds:
            continue

        # now get the right output file
        output_dir = os.path.join(options.run_dir, app_and_args, config)
        if not os.path.isdir(output_dir):
            print(("WARNING the outputdir " + output_dir + " does not exist"))
            continue

        jobId, gpusim_version = specific_jobIds[app_and_args + config]
        jobname = app_and_args + '-' + config + '-' + gpusim_version

        outfile = os.path.join(output_dir, jobname + "." + jobId + '.log')

        if not os.path.isfile(outfile):
            print("WARNING - " + outfile + " does not exist")
            continue

        stat_map[app_and_args + config + 'gpusim_version'] = gpusim_version
        stat_map[app_and_args + config + 'jobid'] = jobId

        # Do a quick 10000-line reverse pass to make sure the simualtion thread finished
        SIM_EXIT_STRING = "GPGPU-Sim: \*\*\* exit detected \*\*\*"
        SIM_INACTIVE_STRING = "GPGPU-Sim: detected inactive GPU simulation thread"
        exit_success = False
        MAX_LINES = 10000
        BYTES_TO_READ = int(250 * 1024 * 1024)
        count = 0
        f = open(outfile)
        fsize = int(os.stat(outfile).st_size)
        if fsize > BYTES_TO_READ:
            f.seek(-BYTES_TO_READ, os.SEEK_END)
        lines = f.readlines()
        for line in reversed(lines):
            count += 1
            if count >= MAX_LINES:
                break
            exit_match = re.match(SIM_EXIT_STRING, line)
            inactive_match = re.match(SIM_INACTIVE_STRING, line)
            exit_match = exit_match or inactive_match
            if exit_match:
                exit_success = True
                break
        del lines
        f.close()

        if not exit_success:
            print("Detected that {0} does not contain a terminating string from GPGPU-Sim. Skip.".format(outfile))
            continue

        stat_found = set()
        files_parsed += 1
        bytes_parsed += os.stat(outfile).st_size

        f = open(outfile)
        lines = f.readlines()

        # print "Parsing File {0}. Size: {1}".format(outfile, millify(os.stat(outfile).st_size))
        # reverse pass cuz the stats are at the bottom
        for line in lines:
            # If we ended simulation due to too many insn - ignore the last kernel launch, as it is no complete.
            # Note: This only appies if we are doing kernel-by-kernel stats
            last_kernel_break = re.match(
                r"GPGPU-Sim: \*\* break due to reaching the maximum cycles \(or instructions\) \*\*", line)
            if last_kernel_break:
                print("NOTE::::: Found Max Insn reached in {0}.".format(outfile))

            for stat_name, token in stats_to_pull.items():
                existance_test = token[0].search(line.rstrip())
                if existance_test:
                    stat_found.add(stat_name)
                    number = existance_test.group(1).strip()
                    number = number.replace(',', 'x')  # avoid conflicts with csv commas

                    if token[1] == 'scalar':
                        stat_map[app_and_args + config + stat_name] = number
                    else:
                        if app_and_args + config + stat_name not in stat_map:
                            stat_map[app_and_args + config + stat_name] = [number]
                        else:
                            stat_map[app_and_args + config + stat_name].append(number)

        f.close()

# print out the csv file
print(('-' * 100))
# just to make sure we print the stats in deterministic order, store keys of the map into a stats_list
stats_list = list(stats_to_pull.keys())

with open(options.outfile, 'w+') as f:
    f.write('pair_str,config,gpusim_version,jobId,' + ','.join(stats_list) + '\n')
    print(('pair_str,config,gpusim_version,jobId,' + ','.join(stats_list)))

    for app_str in apps_and_args:
        for config in configs:
            if app_str + config + 'gpusim_version' not in stat_map:
                continue

            csv_str = app_str + ',' \
                      + config + ',' \
                      + stat_map[app_str + config + 'gpusim_version'] + ',' \
                      + stat_map[app_str + config + 'jobid']

            for stat in stats_list:
                if app_str + config + stat in stat_map:
                    if stats_to_pull[stat][1] == 'scalar':
                        csv_str += ',' + stat_map[app_str + config + stat]
                    else:
                        csv_str += ',[' + ' '.join(stat_map[app_str + config + stat]) + ']'
                else:
                    csv_str += ',' + '0'

            f.write(csv_str + '\n')
            print(csv_str)

print(("Write to file {0}".format(options.outfile)))
print(('-' * 100))

duration = time.time() - start_time

print("Script exec time {0:.2f} seconds. {1} files and {2}B parsed. {3}B/s". \
      format(duration, files_parsed, millify(bytes_parsed),
             millify(float(bytes_parsed) / float(duration))))
