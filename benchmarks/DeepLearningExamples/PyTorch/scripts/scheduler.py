# script to launch serial jobs on n GPUs
import argparse
import subprocess
import time

models = ['resnet', 'gnmt']
tasks = ['train', 'infer']
#metrics = ['comp', 'inst', 'mem', 'time']
metrics = ['time']


def runner(gpus):
    jobs = [[m, t, p] for m in models for t in tasks for p in metrics]

    # initialize the status table
    status = {}
    for g in gpus:
        status[g] = None

    # progress bar
    pbar = tqdm(total=len(jobs))

    idx = 0
    while(idx < len(jobs)):
        for gpu in status:
            # if not free, poll
            if status[gpu]:
                poll = status[gpu].poll()
                if poll != None:
                    # process is done 
                    print('A task is done')
                    status[gpu] = None
                    pbar.update(1)

        for gpu in status:
            # after status update, if free, schedule something
            if not status[gpu]:
                # make cmd to schedule a job
                cmd = ['./run_job.sh'] + jobs[idx] + ['{}'.format(gpu)]
                idx += 1

                # launch job
                print('Launching', cmd)
                status[gpu] = subprocess.Popen(cmd)

        # sleep for 5 min
        sleep(5*60)

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
