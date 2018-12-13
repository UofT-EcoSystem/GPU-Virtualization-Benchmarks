import json
import os
import subprocess
dir_path = os.path.dirname(os.path.realpath(__file__))
envs_path = dir_path + '/../'
root_path = dir_path + '/../../'

def env_paths(args):
    results = []
    if args.benchmark == 'tbd':
        for model in args.models:
            for framework in args.frameworks:
                results.append('_'.join([args.benchmark, model, framework]))
    return results

def manage_env(args):
    if args.command == 'create':
        if not args.force:
            for env_path in env_paths(args):
                if os.path.isdir(envs_path + env_path):
                    # Wrap the whole thing using a try except block?
                    raise IOError('{} environment already existis'.format(env_path))
                else:
                    create_env(env_path)

def create_env(env_path):
    benchmark = env_path.split('_')[0]
    if benchmark == 'tbd':
        benchmark, model, framework = env_path.split('_')

        env_info = json.load(open(dir_path + '/env_info.json'))[benchmark][model][framework]
        virtualenv_flags = env_info['virtualenv_flags']
        requirements = env_info['requirements']
        os.system('virtualenv ' + virtualenv_flags + ' ' + os.path.join([envs_path, env_path]))

        env = envs_path + '/' + env_path

        activate = env + '/bin/activate_this.py'
        execfile(activate, dict(__file__=activate))
        os.system('pip install -r ' + root_path + '/' + requirements)

        print 'created environment ' + env_path


def add_env_cli(subparsers):
    env_parser = subparsers.add_parser('env', help='Environment management')
    env_parser.add_argument('command', choices=['create', 'check', 'activate'],
                            help='Specify the command to perform')

    env_subparsers = env_parser.add_subparsers(title='Available benchmarks', description='', dest='benchmark')

    # TBD benchmark env manager
    tbd_env_parser = env_subparsers.add_parser('tbd', help='TBD benchmark')
    tbd_env_parser.add_argument('-m', '--models', nargs='*',
                            choices=['seq2seq', 'transformer', 'inception'],
                            required=True,
                            help='Specify the models of the benchmarks.')

    tbd_env_parser.add_argument('-f', '--frameworks', choices=['tf', 'mxnet', 'cntk'], nargs='*',
                            default=['tf'],
                            help='Choose the framework that will be used for the model.')

    tbd_env_parser.add_argument('-F', '--force', action='store_true',
                            help='Ignore existing environments.')
