from argparse import ArgumentParser
import yaml
from fairPCA import run
import os.path as osp
from pprint import pprint

class Args:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


if __name__=='__main__':
    arg_parser = ArgumentParser()

    arg_parser.add_argument('--config_dir', type=str, default='EXPS')
    arg_parser.add_argument('--config_name', type=str, default='default')
    arg_parser.add_argument('--pprint', action='store_true')
    config_args = arg_parser.parse_args()

    # Load YAML config
    cfg_dir = osp.abspath(osp.join(osp.dirname(__file__),
                                   config_args.config_dir,
                                   config_args.config_name+'.yaml'))
    args = yaml.load(open(cfg_dir), Loader=yaml.FullLoader)
    
    if config_args.pprint:  pprint(args)
    else:                   print(args)
    print()

    run(Args(args))