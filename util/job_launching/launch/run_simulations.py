import argparse
import os
import sys
import subprocess
import re
import filecmp
import shutil
import glob

import job_launching.launch.common as common
import data.scripts.common.constants as const

args = None


# File system helper functions. #
def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def rm(dir):
    try:
        shutil.rmtree(dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


# Print helper functions. #
def pretty_print(*str):
    print("run_simulations.py >>>", *str)


def parse_args():
    parser = argparse.ArgumentParser("Invoke simulation jobs using job "
                                     "scheduler (Torque/Slurm).")

    parser.add_argument("--app", nargs='+', required=True,
                        help="Benchmark to launch.")

    parser.add_argument("--config", nargs='+', required=True,
                        help="GPGPUSim config name.")

    parser.add_argument("--bench_home",
                        default="/mnt/GPU-Virtualization-Benchmarks"
                                "/benchmarksv2",
                        help="Benchmark home folder.")

    parser.add_argument("--no_launch", action="store_true",
                        help="No real jobs will be launched, but run folder "
                             "will be set up.")

    parser.add_argument("--new_only", action="store_true",
                        help="Launch jobs that do not have existing run "
                             "folder.")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing sim run dir completely.")

    parser.add_argument("--launch_name", default="demo",
                        help="This determines name of the parent folder of the "
                             "run folder. ")

    parser.add_argument("--env", choices=["eco", "vector"], default="eco",
                        help="Choose cluster environment. Eco will invoke "
                             "torque while vector will invoke slurm.")

    global args
    args = parser.parse_args()


def sanity_checks():
    # Check if gpgpusim set up is run
    if str(os.getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN")) != "1":
        sys.exit(
            "ERROR - Please run setup_environment before running this script")

    # Check the existence of required env variables
    if str(os.getenv("GPGPUSIM_ROOT")) == "None":
        exit("\nERROR - Specify GPGPUSIM_ROOT prior to running this script")
    if str(os.getenv("GPGPUSIM_CONFIG")) == "None":
        exit(
            "\nERROR - Specify GPGPUSIM_CONFIG prior to running this script")

    # Check if job scheduler is installed
    if args.env == 'eco':
        # Test for the existence of torque on the system
        if not any([os.path.isfile(os.path.join(p, "qsub")) for p in
                    os.getenv("PATH").split(os.pathsep)]):
            exit(
                "ERROR - Cannot find qsub in PATH... "
                "Is torque installed on this machine?")
    else:
        # Test for the existence of slurm on the system
        if not any([os.path.isfile(os.path.join(p, "sbatch")) for p in
                    os.getenv("PATH").split(os.pathsep)]):
            exit(
                "ERROR - Cannot find sbatch in PATH... "
                "Is slurm installed on this machine?")

    # Check if NVCC is in path
    if not any([os.path.isfile(os.path.join(p, "nvcc")) for p in
                os.getenv("PATH").split(os.pathsep)]):
        exit(
            "ERROR - Cannot find nvcc PATH... "
            "Is CUDA_INSTALL_PATH/bin in the system PATH?")


def copy_so_to_run_dir(launch_run_dir):
    def extract_so_name(so_file):
        p = subprocess.run(["objdump", "-p", so_file], stdout=subprocess.PIPE)
        output = p.stdout.decode('utf-8')

        full_str = re.sub(r".*SONAME\s+([^\s]+).*", r"\1",
                          output.strip().replace("\n", " "))
        full_str = full_str.replace(".so", "")

        # grep the commit half
        match_str = re.search(r"commit-([0-9|a-f]*)_modified.*", full_str)
        commit_id = match_str.group(1)
        return_str = match_str.group(0).replace(commit_id, commit_id[-5:], 1)
        return_str = return_str.replace("-", "_")

        return return_str

    # This function will pull the SO name out of the shared object,
    # which will have current GIT commit number attached.
    def get_cuda_version():
        p = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE)
        output = p.stdout.decode('utf-8')

        cuda_version = re.sub(r".*release (\d+\.\d+).*", r"\1",
                              output.strip().replace("\n", " "))
        os.environ['CUDA_VERSION'] = cuda_version

        return cuda_version

    # Copy .so file into launch run dir from gpgpusim_root/lib/release|debug
    # so that builds don't interfere with running tests
    src_so_path = os.path.join(os.getenv("GPGPUSIM_ROOT"), "lib",
                               os.getenv("GPGPUSIM_CONFIG"),
                               "libcudart.so")

    gpusim_version = extract_so_name(src_so_path)

    so_run_dir = os.path.join(launch_run_dir, "gpgpu-sim-builds",
                              gpusim_version)
    mkdir(so_run_dir)

    cuda_version = get_cuda_version()
    dst_so_path = os.path.join(so_run_dir, "libcudart.so." + cuda_version)

    if not os.path.exists(dst_so_path) or \
            not filecmp.cmp(src_so_path, dst_so_path):
        shutil.copyfile(src_so_path, dst_so_path)

    return gpusim_version, so_run_dir


def gen_job_script(apps, config_name, gpusim_version, so_run_dir,
                   sim_run_dir):
    job_name = "{0}-{1}-{2}".format('-'.join(apps), config_name, gpusim_version)

    job_script = 'torque.sim' if args.env == 'eco' else 'slurm.sim'

    # Build CMAKE defines
    defines = []

    def get_define(_benchmark):
        define_key = _benchmark.split('-')[0]
        return "-D{}=ON".format(define_key)

    for app in apps:
        if app in const.syn_yaml:
            for benchmark in const.syn_yaml[app]:
                defines.append(get_define(benchmark))
        else:
            defines.append(get_define(app))

    # Build input file names
    inputs = ["stream{}.txt".format(idx+1) for idx in range(len(apps))]

    # The job launching threads are doing coarse-grained polling with sleep,
    # So we only need one cpu core to do gpgpusim simulation
    cpu = 1

    replacement_dict = {
        "NAME": job_name,
        "CPU": cpu,
        "GPGPUSIM_ROOT": os.getenv("GPGPUSIM_ROOT"),
        "BENCH_HOME": args.bench_home,
        "LIBPATH": so_run_dir,
        "SUBDIR": sim_run_dir,
        "DEFINES": " ".join(defines),
        "INPUTS": " ".join(inputs),
    }

    this_directory = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(this_directory, job_script)) as f:
        script_text = f.read().strip()

    with open(os.path.join(this_directory, 'job.sim')) as f:
        script_text += '\n' + f.read().strip()

    for entry in replacement_dict:
        script_text = re.sub("REPLACE_" + entry,
                             str(replacement_dict[entry]),
                             script_text)
    return script_text


def gen_input_files(apps):
    result = []
    for app in apps:
        if app in const.syn_yaml:
            # This is a synthetic workload
            inputs = []
            for bench in const.syn_yaml[app]:
                input_line = bench if bench == 'repeat' \
                    else common.get_inputs_from_app(bench, args.bench_home)
                inputs.append(input_line)

            result.append("\n".join(inputs))
        else:
            result.append(common.get_inputs_from_app(app, args.bench_home))

    return result


def gen_gpusim_config(config):
    config_name, base_file, extra_param_text = config

    with open(base_file) as f:
        config_str = f.read()

    # Append customized configs
    config_str += "\n" + extra_param_text

    src_config_dir = os.path.dirname(base_file)

    return config_name, config_str, src_config_dir


def setup_sim_dir(sim_run_dir, job_script_str, config_str,
                  input_str, src_config_dir):
    mkdir(sim_run_dir)

    # Copy config files into sim run dir
    src_files = glob.glob(os.path.join(src_config_dir, '*'))
    for file_to_cp in src_files:
        dst_file = os.path.join(sim_run_dir, os.path.basename(file_to_cp))
        if os.path.isfile(dst_file):
            os.remove(dst_file)

        shutil.copyfile(file_to_cp, dst_file)

    # Write job script
    with open(os.path.join(sim_run_dir, 'job.sim'), 'w+') as f:
        f.write(job_script_str)

    # Write gpgpusim config file
    with open(os.path.join(sim_run_dir, 'gpgpusim.config'), 'w+') as f:
        f.write(config_str)

    # Write input files
    for stream_id, input in enumerate(input_str):
        filename = os.path.join(sim_run_dir, 'stream{}.txt'.format(stream_id+1))
        with open(filename, 'w+') as f:
            f.write(input)


def launch_sim_job(sim_run_dir):
    current_dir = os.getcwd()
    os.chdir(sim_run_dir)

    job_script = os.path.join(sim_run_dir, "job.sim")
    if args.env == 'eco':
        cmd = ["qsub", "-W", "umask=022", job_script]
    else:
        cmd = ["sbatch", job_script]

    p = subprocess.run(cmd, stdout=subprocess.PIPE)

    if p.returncode > 0:
        exit("Error Launching Torque Job")
    else:
        # Parse the torque output for just the numeric ID
        jobid = p.stdout.decode('utf-8').strip()
        print("Job " + jobid + " queued.")

    os.chdir(current_dir)


def main():
    parse_args()

    # Sanity checks:
    sanity_checks()

    parent_run_dir = os.path.join(args.bench_home, 'run')
    # mkdir(parent_run_dir)

    # Create launch run dir
    launch_run_dir = common.get_run_dir(parent_run_dir, args.launch_name)
    mkdir(launch_run_dir)

    # Set up gpusim so path
    gpusim_version, so_run_dir = copy_so_to_run_dir(launch_run_dir)

    # Load yaml defines for apps and configs
    common.load_defined_yamls()

    # Get benchmark pairs
    benchmarks = common.parse_app_list(args.app)

    # Parse configs
    config_tuples = common.gen_configs_from_list(args.config)

    for config in config_tuples:
        # Generate gpgpusim.config for this config
        config_name, config_str, src_config_dir = gen_gpusim_config(config)

        for bench in benchmarks:
            sim_run_dir = os.path.join(launch_run_dir,
                                       '-'.join(bench),
                                       config_name)

            if os.path.exists(sim_run_dir):
                if args.new_only:
                    continue
                if args.overwrite:
                    rm(sim_run_dir)

            # Generate job script for this bench
            job_script_str = gen_job_script(bench, config_name, gpusim_version,
                                            so_run_dir, sim_run_dir)

            # Generate input files
            input_str = gen_input_files(bench)

            # Real deal: create sim run dir and launch sim job
            setup_sim_dir(sim_run_dir, job_script_str, config_str, input_str,
                          src_config_dir)

            if not args.no_launch:
                print('-' * 30)
                pretty_print("Benchmark: ", bench)
                pretty_print("Config: ", config_name)

                launch_sim_job(sim_run_dir)

                print('-' * 30)


if __name__ == '__main__':
    main()
