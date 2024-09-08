import sys
import yaml
import argparse


def parse_external():

    def gather():
        if len(buffer) == 1:
            if i == len(argv) - 1:
                args_external[argv[i][2:]] = True
            else:
                args_external[argv[i-1][2:]] = True
        elif len(buffer) == 2:
            args_external[buffer[0]] = buffer[1]
        elif len(buffer) >= 3:
            args_external[buffer[0]] = buffer[1:]

    buffer = []
    args_external = {}
    argv = sys.argv[1:]
    for i in range(len(argv)):
        if 'local_rank' in argv[i]:
            index = argv[i].rfind('=')
            args_external[argv[i][2: index]] = eval(argv[i][index+1:])
        elif argv[i][:2] == '--':
            gather()
            buffer = [argv[i][2:]]
        else:
            try:
                value = eval(argv[i])
            except:
                value = argv[i]
            buffer.append(value)
    gather()
    return args_external


def merge_args(args):
    if isinstance(args, tuple):
        assert len(args) == 2
        args = args[0]
    if args.cfg is not None:
        # command > config file > defaults
        with open(args.cfg, 'r') as f:
            preset_cfgs = yaml.safe_load(f)
        args_defaults = parser.parse_args([])
        args_defaults.__dict__.update(preset_cfgs)
        args_external = parse_external()
        args_defaults.__dict__.update(args_external)
        args = args_defaults
    return args


parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', default='CIFAR100', type=str, choices=['PMNIST', 'RMNIST', 'CIFAR100',
                                                                        'TinyImageNet', 'ImageNet100', 'ImageNet1000'])
parser.add_argument('--datadir', default='data', type=str, help='root of dataset')
parser.add_argument('--scenario', default='task', type=str, choices=['task', 'domain', 'class'])
parser.add_argument('--order', default=None, type=int, help='index of classes order in .yaml file')
parser.add_argument('--init', default=None, type=int, help='number of classes in the first task')
parser.add_argument('--tasks', default=20, type=int, help='number of tasks needed to learn')
# General setting
parser.add_argument('--scheme', default='FineTune', type=str, help='')
parser.add_argument('--dropout', default=0., type=float)
parser.add_argument('--optim', default='SGD', type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--decay', default=0.0, type=float)
parser.add_argument('--momentum', default=0.7, type=float)
parser.add_argument('--steps', default=None, nargs='+', type=int)
parser.add_argument('--gamma', default=1., type=float)
parser.add_argument('--memory', default=0, type=int)
parser.add_argument('--mcat', action='store_true')
parser.add_argument('--bs', default=10, type=int)
parser.add_argument('--mbs', default=10, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--online', action='store_true')
# Hyperparameters
parser.add_argument('--alpha', default=0., type=float)
parser.add_argument('--beta', default=0., type=float)
parser.add_argument('--delta', default=0., type=float)
parser.add_argument('--epsilon', default=0., type=float)
parser.add_argument('--zeta', default=0., type=float)
parser.add_argument('--eta', default=0., type=float)
parser.add_argument('--theta', default=0., type=float)
parser.add_argument('--lambd', default=0., type=float)
parser.add_argument('--sigma', default=0., type=float)
parser.add_argument('--tau', default=0., type=float)
parser.add_argument('--omega', default=0., type=float)
# Device
parser.add_argument('--local_rank', default='0', type=int)
parser.add_argument('--mode', default='GPU', choices=['GPU', 'DP', 'DDP'])
parser.add_argument('--gpuid', default=0, nargs='+', type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--FAST', action='store_true')
parser.add_argument('--threads', default=None, type=int)
parser.add_argument('--pin_memory', action='store_true')
parser.add_argument('--seed', default=1993, type=int)
# Input
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
# Output
parser.add_argument('--logdir', default='log/', type=str)
parser.add_argument('--name', default='test', type=str)
parser.add_argument('--note', default='', type=str)
parser.add_argument('--snapshot', action='store_true')
parser.add_argument('--opt', default=[], nargs='+', help="choose any number of modes from ['summary', 'slient', "
                                                         "'time_memory', 'bark']")
parser.add_argument('--grid', default='', type=str)
parser.add_argument('--debug', action='store_true', help="this operation will expose private variables for easier "
                                                         "debugging")
args = parser.parse_known_args()
args = merge_args(args)
