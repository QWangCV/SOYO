import json
import yaml
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_yaml(args.config)
    args = vars(args)   # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from yaml
    train(args)

def load_yaml(settings_path):
    with open(settings_path, 'r', encoding='utf-8') as file:
        param = yaml.safe_load(file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='DIL')
    parser.add_argument('--config', type=str, default='config/xxx.yaml', help='')
    parser.add_argument('--device', '-g', nargs='+', type=int, help='')
    
    return parser

if __name__ == '__main__':
    main()
