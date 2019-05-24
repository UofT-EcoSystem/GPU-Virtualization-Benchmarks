#!/usr/bin/env python

import argparse
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../tbd-core/cli')

from tbd_cli import add_tbd_cli, run_tbd
from envs import add_env_cli, manage_env

def main_cli():

    main_parser = argparse.ArgumentParser(description='benchmark collection')
    subparsers = main_parser.add_subparsers(title='Available benchmarks', description='', dest='subparser')

    add_tbd_cli(subparsers)
    add_env_cli(subparsers)

    args = main_parser.parse_args()

    benchmark = {
        'tbd': run_tbd,
        'env': manage_env,
    }

    return benchmark[args.subparser](args)
