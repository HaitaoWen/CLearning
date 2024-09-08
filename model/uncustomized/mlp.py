import operator
from collections import OrderedDict
from typing import Iterable
from functools import reduce

from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, Module
from torch.nn.functional import log_softmax
import collections
from itertools import repeat


def ntuple(x, n):
    if isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) != 1:
            assert len(x) == n, "Expected length %d, but found %d" % (n, len(x))
            return x
        else:
            x = x[0]
    return tuple(repeat(x, n))


class MLP(Module):
    def __init__(self, args, num_classes=100):
        super().__init__()

        depth = args.depth
        widths = args.width
        input_size = args.input
        output_size = num_classes
        bias = args.bias
        batch_norm = args.batch_norm

        dropout = 0
        widths = ntuple(widths, depth)

        layers = OrderedDict()
        previous_size = reduce(operator.mul, input_size) if isinstance(input_size, Iterable) else input_size
        non_lin = ReLU()
        for i, width in enumerate(widths):
            layer = OrderedDict()
            layer["linear"] = Linear(previous_size, width, bias=bias)
            previous_size = width
            if batch_norm:
                layer["batch_norm"] = BatchNorm1d(width)
            layer["non_lin"] = non_lin
            if dropout > 0:
                layer["dropout"] = Dropout(float(dropout))
            layers["layer_{i}".format(i=i)] = Sequential(layer)
        self.body = Sequential(layers)
        self.fc = Linear(previous_size, output_size, bias=bias)

    def forward(self, data):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        data = self.body(data)
        data = self.fc(data)
        return data
