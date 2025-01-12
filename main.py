import json
import argparse

import pandas as pd
from trainer import train


def main():
    args = {}
    dynamic_args = setup_parser().parse_args()
    param = load_json(dynamic_args.config)
    args.update(param)  # Add parameters from json first, and then add parameters from dynamic args
    dynamic_args = vars(dynamic_args)  # Converting argparse Namespace to a dict.
    args.update(dynamic_args) # update the dynamic_args after json args, so dynamic args' priority are higher
    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='/data/syj/MAR/exps/args.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
