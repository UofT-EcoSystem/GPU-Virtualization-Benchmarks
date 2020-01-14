#!/usr/bin/env python

from optparse import OptionParser
import os
import subprocess
import sys
import re
import shutil
import glob
import datetime
import common
import filecmp

this_directory = os.path.dirname(os.path.realpath(__file__))


def parse_run_simulations_options():
    parser = OptionParser()
    parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                      help="A file with all benchmark pairs to run. See apps/define-*.yml for " + \
                           "the benchmark suite names.",
                      default="apps/default.pair")
    parser.add_option("-C", "--configs_list", dest="configs_list",
                      help="a comma seperated list of configs to run. See configs/define-*.yml for " + \
                           "the config names.",
                      default="GTX480")
    parser.add_option("-E", "--benchmark-root", dest="benchmark_root",
                      help="ABSOLUTE path of the benchmarkv2 root directory.",
                      default="/mnt/GPU-Virtualization-Benchmarks/benchmarksv2/")
    parser.add_option("-p", "--benchmark_exec_prefix", dest="benchmark_exec_prefix",
                      help="When submitting the job to torque this string" + \
                           " is placed before the command line that runs the benchmark. " + \
                           " Useful when wanting to run valgrind.", default="")
    parser.add_option("-n", "--no_launch", dest="no_launch", action="store_true",
                      help="When set, no torque jobs are launched.  However, all" + \
                           " the setup for running is performed. ie, the run" + \
                           " directories are created and are ready to run." + \
                           " This can be useful when you want to create a new" + \
                           " configuration, but want to test it locally before " + \
                           " launching a bunch of jobs.")
    parser.add_option("-s", "--so_dir", dest="so_dir",
                      help="Point this to the directory that your .so is stored in. If nothing is input here - " + \
                           "the scripts will assume that you are using the so built in GPGPUSIM_ROOT.",
                      default="")
    parser.add_option("-N", "--launch_name", dest="launch_name", default="",
                      help="Pass if you want to name the launch. This will determine the name of the logfile.\n" + \
                           "If you do not name the file, it will just use the current date/time.")

    (options, args) = parser.parse_args()

    # Parser seems to leave some whitespace on the options, getting rid of it
    options.configs_list = options.configs_list.strip()
    options.benchmark_list = options.benchmark_list.strip()
    options.benchmark_root = options.benchmark_root.strip()
    options.launch_name = options.launch_name.strip()

    # optional
    options.so_dir = options.so_dir.strip()
    options.benchmark_exec_prefix = options.benchmark_exec_prefix.strip()

    # derived paths
    parent_run_dir = os.path.join(options.benchmark_root, 'run')
    if not os.path.exists(parent_run_dir):
        os.makedirs(parent_run_dir)

    options.run_directory = common.get_run_dir(parent_run_dir, options.launch_name)
    options.log_directory = common.get_log_dir(parent_run_dir)

    return options, args


# This function will pull the SO name out of the shared object,
# which will have current GIT commit number attached.
def extract_so_name(so_file):
    objdump_out_filename = this_directory + "so_objdump_out.{0}.txt".format(os.getpid())
    objdump_out_file = open(objdump_out_filename, 'w+')
    subprocess.call(["objdump", "-p", so_file], stdout=objdump_out_file)
    objdump_out_file.seek(0)

    full_str = re.sub(r".*SONAME\s+([^\s]+).*", r"\1", objdump_out_file.read().strip().replace("\n", " "))
    full_str = full_str.replace(".so", "")

    # grep the commit half
    match_str = re.search(r"commit-([0-9|a-f]*)_modified.*", full_str)
    commit_id = match_str.group(1)
    return_str = match_str.group(0).replace(commit_id, commit_id[-5:], 1)
    return_str = return_str.replace("-", "_")

    objdump_out_file.close()
    os.remove(objdump_out_filename)
    return return_str


def get_cuda_version():
    p = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE)
    output = p.stdout.decode('utf-8')

    cuda_version = re.sub(r".*release (\d+\.\d+).*", r"\1", output.strip().replace("\n", " "))
    os.environ['CUDA_VERSION'] = cuda_version

    return cuda_version


#######################################################################################
# Class the represents each configuration you are going to run
# For example, if your sweep file has 2 entries 32k-L1 and 64k-L1 there will be 2
# ConfigurationSpec classes and the run_subdir name for each will be 32k-L1 and 64k-L1
# respectively
class ConfigurationSpec:
    #########################################################################################
    # Public Interface methods
    #########################################################################################
    # Class is constructed with a single line of text from the sweep_param file
    def __init__(self, config_params):
        (self.config_name, self.base_file, self.extra_param_text) = config_params

    def my_print(self):
        print("Run Subdir = " + self.config_name)
        print("Base config file = " + self.base_file)
        print("Parameters = " + self.extra_param_text)

    def run(self):
        for pair in ConfigurationSpec.benchmarks:
            pair_str = '-'.join(pair)
            this_run_dir = os.path.join(ConfigurationSpec.run_dir, pair_str, self.config_name)

            self.__setup_run_directory(this_run_dir)
            self.__text_replace_torque_sim(this_run_dir, pair)

            if ConfigurationSpec.no_launch:
                continue

            # Submit the job to torque and dump the output to a file
            saved_dir = os.getcwd()
            os.chdir(this_run_dir)
            cmd = ["qsub", "-W", "umask=022", os.path.join(this_run_dir, "torque.sim")]

            p = subprocess.run(cmd, stdout=subprocess.PIPE)

            if p.returncode > 0:
                exit("Error Launching Torque Job")
            else:
                # Parse the torque output for just the numeric ID
                torque_jobid = p.stdout.decode('utf-8').strip()
                print(("Job " + torque_jobid + " queued (" + pair_str + ", " + self.config_name + ")"))

            os.chdir(saved_dir)

            if len(torque_jobid) > 0:
                # Dump the benchmark description to the logfile
                if not os.path.exists(ConfigurationSpec.log_dir):
                    # In the very rare case that concurrent builds try to make the directory at the same time
                    # (after the test to os.path.exists -- this has actually happened...)
                    try:
                        os.makedirs(ConfigurationSpec.log_dir)
                    except:
                        pass

                now_time = datetime.datetime.now()
                day_string = now_time.strftime("%y.%m.%d-%A")
                time_string = now_time.strftime("%H:%M:%S")

                import socket
                hostname = socket.gethostname()

                log_name = "sim_log.{0}.{1}".format(hostname, ConfigurationSpec.launch_name)

                with open(os.path.join(ConfigurationSpec.log_dir, log_name + "." + day_string + ".txt"), 'a') \
                        as logfile:
                    print("%s %6s %-22s %-25s %s" %
                          (time_string,
                           torque_jobid,
                           pair_str,
                           self.config_name,
                           ConfigurationSpec.version_string),
                          file=logfile)

    #########################################################################################
    # Internal utility methods
    #########################################################################################
    # copies all the necessary files to the run directory
    # update gpgpusim.config with the extra config params
    def __setup_run_directory(self, this_run_dir):
        if not os.path.isdir(this_run_dir):
            os.makedirs(this_run_dir)

        # copy everything over from the config folder
        files_to_copy_to_run_dir = glob.glob(os.path.dirname(self.base_file) + "/*")

        for file_to_cp in files_to_copy_to_run_dir:
            new_file = os.path.join(this_run_dir, os.path.basename(file_to_cp))
            if os.path.isfile(new_file):
                os.remove(new_file)
            shutil.copyfile(file_to_cp, new_file)

        config_text = open(os.path.join(this_run_dir, 'gpgpusim.config')).read()
        config_text += "\n" + self.extra_param_text

        open(os.path.join(this_run_dir, "gpgpusim.config"), 'w').write(config_text)

    # replaces all the "REAPLCE_*" strings in the torque.sim file
    def __text_replace_torque_sim(self, this_run_dir, pair):
        pair_str = '-'.join(pair)

        # Test the existence of required env variables
        if str(os.getenv("GPGPUSIM_ROOT")) == "None":
            exit("\nERROR - Specify GPGPUSIM_ROOT prior to running this script")
        if str(os.getenv("GPGPUSIM_CONFIG")) == "None":
            exit("\nERROR - Specify GPGPUSIM_CONFIG prior to running this script")

        # do the text replacement for the torque.sim file

        if os.getenv("TORQUE_QUEUE_NAME") is None:
            queue_name = "batch"
        else:
            queue_name = os.getenv("TORQUE_QUEUE_NAME")

        _input_1 = common.get_inputs_from_app(pair[0])
        _app_1_cmake = _input_1.split(' ')[0]

        if len(pair) > 1:
            _valid_app_2 = 'true'
            _app_2 = pair[1]
            _input_2 = common.get_inputs_from_app(pair[1])
            _app_2_cmake = _input_2.split(' ')[0]

            _ppn = "4"
        else:
            _valid_app_2 = 'false'
            _app_2 = 'dont_care'
            _input_2 = 'dont_care'
            _app_2_cmake = 'dont_care'
            _ppn = "3"

        replacement_dict = {"NAME": pair_str + '-' + self.config_name + '-' + ConfigurationSpec.version_string,
                            "NODES": "1",
                            "PPN": _ppn,
                            "QUEUE_NAME": queue_name,
                            "GPGPUSIM_ROOT": os.getenv("GPGPUSIM_ROOT"),
                            "BENCH_HOME": ConfigurationSpec.benchmark_root,
                            "LIBPATH": ConfigurationSpec.so_dir,
                            "SUBDIR": this_run_dir,
                            "APP_1": pair[0],
                            "SHORT_APP_1": _app_1_cmake,
                            "VALID_APP_2": _valid_app_2,
                            "APP_2": _app_2,
                            "SHORT_APP_2": _app_2_cmake,
                            "INPUT_1": _input_1,
                            "INPUT_2": _input_2,
                            "PATH": os.getenv("PATH"),
                            }

        torque_text = open(os.path.join(this_directory, "torque.sim")).read().strip()
        for entry in replacement_dict:
            torque_text = re.sub("REPLACE_" + entry,
                                 str(replacement_dict[entry]),
                                 torque_text)
        open(os.path.join(this_run_dir, "torque.sim"), 'w').write(torque_text)


# -----------------------------------------------------------
# main script start
# -----------------------------------------------------------

def main():
    (options, args) = parse_run_simulations_options()

    # 0. Environment checks
    # Check if gpgpusim setup is run
    if str(os.getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN")) != "1":
        sys.exit("ERROR - Please run setup_environment before running this script")

    # Test for the existence of torque on the system
    if not any([os.path.isfile(os.path.join(p, "qsub")) for p in os.getenv("PATH").split(os.pathsep)]):
        exit("ERROR - Cannot find qsub in PATH... Is torque installed on this machine?")

    # Check if NVCC is in path
    if not any([os.path.isfile(os.path.join(p, "nvcc")) for p in os.getenv("PATH").split(os.pathsep)]):
        exit("ERROR - Cannot find nvcc PATH... Is CUDA_INSTALL_PATH/bin in the system PATH?")

    cuda_version = get_cuda_version()

    # 1. Make run directory
    if options.run_directory == "":
        options.run_directory = os.path.join(this_directory, "../../sim_run_%s" % cuda_version)

    if not os.path.isdir(options.run_directory):
        try:
            os.makedirs(options.run_directory)
        except:
            print(("Failed to create run directory %s" % options.run_directory))
            exit(1)

    # 2. Copy .so file into run dir
    # Let's copy out the .so file so that builds don't interfere with running tests
    # If the user does not specify a so file, then use the one in the git repo and copy it out.
    options.so_dir = common.dir_option_test(
        options.so_dir, os.path.join(os.getenv("GPGPUSIM_ROOT"), "lib", os.getenv("GPGPUSIM_CONFIG")),
        this_directory)
    so_path = os.path.join(options.so_dir, "libcudart.so")
    version_string = extract_so_name(so_path)
    running_so_dir = os.path.join(options.run_directory, "gpgpu-sim-builds", version_string)
    if not os.path.exists(running_so_dir):
        # In the very rare case that concurrent builds try to make the directory at the same time
        # (after the test to os.path.exists -- this has actually happened...)
        try:
            os.makedirs(running_so_dir)
        except:
            pass

    src = so_path
    dst = os.path.join(running_so_dir, "libcudart.so." + cuda_version)
    if not os.path.exists(dst) or not filecmp.cmp(src, dst):
        shutil.copyfile(src, dst)
    options.so_dir = running_so_dir

    # 3. Load yaml defines
    common.load_defined_yamls()

    # 4. Get benchmark pairs
    benchmarks = common.parse_app_list(options.benchmark_list)

    # 5. Parse configs
    config_tuples = common.gen_configs_from_list(options.configs_list)

    print(("Running Simulations with GPGPU-Sim built from \n{0}\n ".format(version_string) +
           "\nUsing configs: " + options.configs_list +
           "\nBenchmark: " + options.benchmark_list))

    # 6. Launch jobs
    # Static variables for ConfigureSpec: benchmarks, run dir, so dir, job name, no_launch (bool)
    ConfigurationSpec.benchmarks = benchmarks
    ConfigurationSpec.benchmark_root = options.benchmark_root
    ConfigurationSpec.run_dir = options.run_directory
    ConfigurationSpec.log_dir = options.log_directory
    ConfigurationSpec.so_dir = options.so_dir
    ConfigurationSpec.version_string = version_string
    ConfigurationSpec.launch_name = options.launch_name
    ConfigurationSpec.no_launch = options.no_launch

    for config in config_tuples:
        config_spec = ConfigurationSpec(config)

        config_spec.my_print()
        config_spec.run()


if __name__ == '__main__':
    main()
