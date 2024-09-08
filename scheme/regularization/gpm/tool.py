import numpy as np
from copy import deepcopy


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if epoch == 1:
            param_group['lr'] = args.lr
        else:
            param_group['lr'] *= args.gamma
