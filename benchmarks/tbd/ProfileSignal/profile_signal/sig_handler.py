import os
import signal
import numba.cuda as cuda

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def start_profiling(sig, frame):
    print '=== START PROFILING ==='
    cuda.profile_start()

def stop_profiling(sig, frame):
    print '=== STOP PROFILING ==='
    cuda.profile_stop()
    exit()

def profiling_period():
    signal.signal(signal.SIGUSR1, start_profiling)
    print bcolors.OKGREEN + '[Signal Handler] [{}] Waiting for SIGUSR1 to start.'.format(os.getpid()) + bcolors.ENDC
    signal.pause()
    signal.signal(signal.SIGUSR2, stop_profiling)
    print bcolors.OKGREEN + '[Signal Handler] [{}] Started profiling, waiting for SIGUSR2 to stop.'.format(os.getpid()) + bcolors.ENDC
