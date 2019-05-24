import json
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

from download_datasets import downloader_map

def add_tbd_cli(subparsers):
    """Add TBD to the main CLI
    
    Args:
        subparsers: The sub-parsers of the main CLI
    """
    tbd_parser = subparsers.add_parser('tbd', help='TBD benchmark')

    tbd_parser.add_argument('-m', '--models', nargs='*',
                            choices=['seq2seq', 'transformer', 'inception'],
                            required=True,
                            help='Specify the models to be profiled.')
    tbd_parser.add_argument('-M', '--profile-metrics', nargs='*',
                            help='Specify the metrics to be profiled. Use the command `nvprof --query-metrics` to see the full list of available metrics for your device.')
    tbd_parser.add_argument('-P', '--profile-mode-off', action='store_true',
                            help='Disable profile mode.')
    tbd_parser.add_argument('-c', '--concurrent', action='store_true',
                            help='Profile in concurrent mode.')
    tbd_parser.add_argument('-f', '--frameworks', choices=['tf', 'mxnet', 'cntk'], nargs='*', default=['tf'],
                            help='Choose the framework that will be used for the model.')
    tbd_parser.add_argument('-b', '--batch-size', type=int, default=16,
                            help='Specify the batch size of the training procedure.')
    tbd_parser.add_argument('-o', '--output', default='profile_result',
                            help='Output nvvp file prefix.')
    tbd_parser.add_argument('-d', '--output-directory', default='measurements',
                            help='Output nvvp file directory.')
    tbd_parser.add_argument('--download', action='store_true',
                            help='Download the dataset')
    tbd_parser.add_argument('-e', '--environment', default=None,
                            help='Specify the environment to use.')

model_dir_map = {
    'seq2seq': 'MachineTranslation-Seq2Seq',
    'transformer': 'MachineTranslation-Transformer',
    'inception': 'ImageClassification-Inception_v3',
}

framework_name_map = {
    'tf': 'TensorFlow',
    'mxnet': 'MXNet',
    'cntk': 'CNTK',
}

def run_tbd(args):
    for model in args.models:
        for framework in args.frameworks:
            if args.download:
                download(model, framework, args)
            else:
                run_model(model, framework, args)

inception_tf = ' --train_dir={train_dir} --dataset_dir={dataset_dir} --model_name=inception_v3 --optimizer=sgd --batch_size={batch_size} --dataset_name=cifar10 --learning_rate=0.1 --learning_rate_decay_factor=0.1 --num_epochs_per_decay=30 --weight_decay=0.0001 '

def download(model, framework, args):
    model_dir = os.path.join(dir_path, '..', model_dir_map[model], framework_name_map[framework])
    dataset_dir = os.path.join(model_dir, 'dataset')

    info = json.load(open(os.path.join(dir_path, 'tbd_info.json')))[model][framework]
    get_dataset_command = info['get_dataset'].format(**{
        'model_dir': model_dir,
        'dataset_dir': dataset_dir,
    })
    print get_dataset_command
    os.system(get_dataset_command)
    

def run_model(model, framework, args):
    model_dir = os.path.join(dir_path, '..', model_dir_map[model], framework_name_map[framework])
    dataset_dir = os.path.join(model_dir, 'dataset')

def run_model(model, framework, args):
    info = json.load(open(os.path.join(dir_path, 'tbd_info.json')))[model][framework]

    model_dir = os.path.join(dir_path, '..', model_dir_map[model], framework_name_map[framework])
    trainer = os.path.join(model_dir, 'source', info['trainer'])
    train_dir = os.path.join(model_dir, 'log')
    dataset_dir = os.path.join(model_dir, 'dataset')

    if args.profile_mode_off:
        prefix = ''
        suffix = ''
    else:
        if args.concurrent:
            prefix = '/usr/local/cuda-9.0/bin/nvprof --profile-from-start off --export-profile {}/{}_{}_{}_{}.nvvp -f --print-summary'.format(args.output_directory, args.output, model, framework, os.getpid())
            suffix = ' --nvprof_on=True --concurrent=True'
        else:
            if not args.profile_metrics:
                prefix = '/usr/local/cuda-9.0/bin/nvprof --profile-from-start off --export-profile {}/{}_{}_{}.nvvp -f --print-summary'.format(args.output_directory, args.output, model, framework)
                suffix = ' --nvprof_on=True'
            else:
                prefix = '/usr/local/cuda-9.0/bin/nvprof --profile-from-start off --export-profile {}/{}_{}_{}_{}.nvvp -f --metrics {}--print-summary'.format(args.output_directory, args.output, model, framework, '_'.join(args.metrics), ' '.join(metrics))
                suffix = ' --nvprof_on=True'

    flags = info['config']
    flags_string = ' '.join('--' + k + '=' + v for k, v in flags.items())
    command_type = info['command_type']

    if args.environment == None:
        environment = '_'.join(['tbd', model, framework])
    else:
        environment = args.environment
    #activate = os.path.join(dir_path, '..', '..', 'envs', environment, 'bin', 'activate_this.py')
    #execfile(activate, dict(__file__=activate))


    if command_type == 'file':
        command = prefix + ' python ' + trainer + ' ' + flags_string + suffix
    elif command_type == 'module':
        os.chdir(trainer)
        trainer_module = info['trainer_module']
        command = prefix + ' python -m ' + trainer_module + ' ' + flags_string + suffix
    print command.format(**{'batch_size': args.batch_size, 'train_dir': train_dir, 'model_dir': model_dir, 'dataset_dir': dataset_dir})
    os.system(command.format(**{'batch_size': args.batch_size, 'train_dir': train_dir, 'model_dir': model_dir, 'dataset_dir': dataset_dir}))
