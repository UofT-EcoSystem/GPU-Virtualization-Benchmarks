# script to launch serial jobs on n GPUs
import argparse
import subprocess
import time
from tqdm import tqdm


models = ['resnet', 'gnmt']
tasks = ['train', 'infer']
metrics = ['comp', 'inst', 'mem', 'time']


def runner(gpus):
    jobs = [[m, t, p] for m in models for t in tasks for p in metrics]

    # initialize the status table
    status = {}
    for g in gpus:
        status[g] = None

    # progress bar
    pbar = tqdm(total=len(jobs))

    dispatch_idx = 0
    num_complete = 0
    while(num_complete != len(jobs)):
        for gpu in status:
            # if not free, poll
            if status[gpu]:
                poll = status[gpu].poll()
                if poll != None:
                    # process is done 
                    print('A task is done')
                    status[gpu] = None
                    num_complete += 1
                    pbar.update(1)

        if dispatch_idx < len(jobs):
            for gpu in status:
                # after status update, if free, schedule something
                if not status[gpu]:
                    # make cmd to schedule a job
                    cmd = ['./run_job.sh'] + jobs[dispatch_idx] + ['{}'.format(gpu)]
                    dispatch_idx += 1

                    # launch job
                    print('Launching', cmd)
                    status[gpu] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # sleep for 5 min
        time.sleep(10)

    pbar.close()

def main():
    parser = argparse.ArgumentParser('Launch serial jobs on multi-GPUs.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--gpus', required=True, nargs='+', 
            help='GPU ids to run jobs on')

    args = parser.parse_args()

    runner(args.gpus)

if __name__ == '__main__':
    main()
