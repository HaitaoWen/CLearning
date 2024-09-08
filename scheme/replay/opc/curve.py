import math
import torch
import numpy as np
import torch.nn as nn
from functools import reduce


class Line(nn.Module):
    def __init__(self, start, stop):
        super(Line, self).__init__()
        for p1, p2 in zip(start, stop):
            assert p1.shape == p2.shape

        self.stop = stop
        self.start = start

    def tangent(self, t):
        parameters = []
        for p1, p2 in zip(self.start, self.stop):
            parameters.append(p2 - p1)
        return parameters

    def forward(self, t):
        parameters = []
        for p1, p2 in zip(self.start, self.stop):
            parameters.append(p1 * (1 - t) + p2 * t)
        return parameters


class MIXLineFourier(nn.Module):
    def __init__(self, start, stop, order=5, boundary=1.):
        super(MIXLineFourier, self).__init__()
        for p1, p2 in zip(start, stop):
            assert p1.shape == p2.shape

        self.stop = stop
        self.start = start
        self.order = order
        self.cos_coef = None
        self.sin_coef = None
        self.boundary = boundary
        self.pi = torch.tensor(math.pi)
        self.displacement = torch.tensor([1]).unsqueeze(1)
        times = torch.tensor([4 * float(n) + 1 for n in range(self.order)]).unsqueeze(dim=0)
        self.register_buffer('frequency', torch.mm(2 * self.pi / (4 * self.displacement), times))
        self.initialize()

    def initialize(self):
        # coefficient = torch.randn(2, 1, self.order)
        # cos_coef = self.normalize_coef(coefficient[0])
        # sin_coef = self.normalize_coef(coefficient[1])
        cos_coef = torch.zeros(1, self.order)
        sin_coef = torch.zeros(1, self.order)
        if self.cos_coef is None or self.sin_coef is None:
            self.cos_coef = nn.Parameter(cos_coef, requires_grad=True)
            self.sin_coef = nn.Parameter(sin_coef, requires_grad=True)
        else:
            self.cos_coef.data = cos_coef.to(self.cos_coef.device)
            self.sin_coef.data = sin_coef.to(self.sin_coef.device)

    def normalize_coef(self, coef):
        mean_old = coef.mean(dim=1).unsqueeze(dim=1)
        coef = coef - mean_old
        maximum = coef.abs().max(dim=1)[0].unsqueeze(dim=1)
        times = torch.rand([coef.shape[0], 1])
        coef = coef / maximum * self.boundary * times
        # s = coef.sum(dim=1)
        # m = coef.max(dim=1)[0]
        return coef

    def correct_shape(self, parameters):
        for index in range(len(parameters)):
            param, stop_ = parameters[index], self.stop[index]
            assert param.numel() == stop_.numel()
            if param.shape != stop_.shape:
                parameters[index] = param.reshape(stop_.shape)

    def tangent(self, t):
        sin_bias = torch.sin(self.frequency * t)
        cos_bias = torch.cos(self.frequency * t)
        start_coef = -torch.sum(torch.mul(torch.mul(self.cos_coef, self.frequency), sin_bias), dim=1) - 1
        stop__coef =  torch.sum(torch.mul(torch.mul(self.sin_coef, self.frequency), cos_bias), dim=1) + 1
        parameters = []
        for p1, p2 in zip(self.start, self.stop):
            parameters.append(p1 * start_coef + p2 * stop__coef)
        self.correct_shape(parameters)
        return parameters

    def forward(self, t):
        cos_bias = torch.cos(self.frequency * t)
        sin_bias = torch.sin(self.frequency * t)
        start_coef = torch.sum(torch.mul(self.cos_coef, cos_bias), dim=1) + (1 - t)
        stop__coef = torch.sum(torch.mul(self.sin_coef, sin_bias), dim=1) + t
        parameters = []
        for p1, p2 in zip(self.start, self.stop):
            parameters.append(p1 * start_coef + p2 * stop__coef)
        self.correct_shape(parameters)
        return parameters


class MIXLineFourierLayer(nn.Module):
    def __init__(self, start, stop, order=5, boundary=1.):
        super(MIXLineFourierLayer, self).__init__()
        for p1, p2 in zip(start, stop):
            assert p1.shape == p2.shape

        self.eps = 1e-2
        self.stop = stop
        self.start = start
        self.order = order
        self.cos_coef = None
        self.sin_coef = None
        self.boundary = boundary
        self.pi = torch.tensor(math.pi)
        self.displacement = torch.tensor([1] * len(start)).unsqueeze(1)
        times = torch.tensor([4 * float(n) + 1 for n in range(self.order)]).unsqueeze(dim=0)
        self.register_buffer('frequency', torch.mm(2 * self.pi / (4 * self.displacement), times))
        self.initialize()

    def initialize(self):
        # coefficient = torch.randn(2, len(self.start), self.order)
        # cos_coef = self.normalize_coef(coefficient[0])
        # sin_coef = self.normalize_coef(coefficient[1])
        cos_coef = torch.zeros(len(self.start), self.order)
        sin_coef = torch.zeros(len(self.start), self.order)
        if self.cos_coef is None or self.sin_coef is None:
            self.cos_coef = nn.Parameter(cos_coef, requires_grad=True)
            self.sin_coef = nn.Parameter(sin_coef, requires_grad=True)
        else:
            self.cos_coef.data = cos_coef.to(self.cos_coef.device)
            self.sin_coef.data = sin_coef.to(self.sin_coef.device)

    def normalize_coef(self, coef):
        mean_old = coef.mean(dim=1).unsqueeze(dim=1)
        coef = coef - mean_old
        maximum = coef.abs().max(dim=1)[0].unsqueeze(dim=1)
        times = torch.rand([coef.shape[0], 1])
        coef = coef / maximum * self.boundary * times
        # s = coef.sum(dim=1)
        # m = coef.max(dim=1)[0]
        return coef

    def correct_shape(self, parameters):
        for index in range(len(parameters)):
            param, stop_ = parameters[index], self.stop[index]
            assert param.numel() == stop_.numel()
            if param.shape != stop_.shape:
                parameters[index] = param.reshape(stop_.shape)

    def tangent(self, t):
        sin_bias = torch.sin(self.frequency * t)
        cos_bias = torch.cos(self.frequency * t)
        start_coef = -torch.sum(torch.mul(torch.mul(self.cos_coef, self.frequency), sin_bias), dim=1) - 1
        stop__coef =  torch.sum(torch.mul(torch.mul(self.sin_coef, self.frequency), cos_bias), dim=1) + 1
        parameters = []
        for p1, c1, p2, c2 in zip(self.start, start_coef, self.stop, stop__coef):
            parameters.append(p1 * c1 + p2 * c2)
        self.correct_shape(parameters)
        return parameters

    def forward(self, t):
        cos_bias = torch.cos(self.frequency * t)
        sin_bias = torch.sin(self.frequency * t)
        start_coef = torch.sum(torch.mul(self.cos_coef, cos_bias), dim=1) + (1 - t)
        stop__coef = torch.sum(torch.mul(self.sin_coef, sin_bias), dim=1) + t
        parameters = []
        for p1, c1, p2, c2 in zip(self.start, start_coef, self.stop, stop__coef):
            parameters.append(p1 * c1 + p2 * c2)
        self.correct_shape(parameters)
        return parameters


class MIXLineFourierPoint(nn.Module):
    def __init__(self, start, stop, order=5, boundary=1.):
        super(MIXLineFourierPoint, self).__init__()
        for p1, p2 in zip(start, stop):
            assert p1.shape == p2.shape

        self.order = order
        self.cos_coef = None
        self.sin_coef = None
        self.boundary = boundary
        self.pi = torch.tensor(math.pi)
        self.start, _ = self.flatten(start)
        self.stop, self.shapes = self.flatten(stop)
        self.displacement = torch.tensor([1.] * self.start.shape[1]).unsqueeze(0)
        times = torch.tensor([4 * float(n) + 1 for n in range(self.order)]).unsqueeze(dim=1)
        self.register_buffer('frequency', torch.mm(times, 2 * self.pi / (4 * self.displacement)))
        self.initialize()

    @staticmethod
    def flatten(params):
        p, shapes = [], []
        for param in params:
            shapes.append(param.shape)
            p.append(param.view(1, -1))
        return torch.cat(p, dim=1), shapes

    def warp(self, params):
        p = []
        begin, end = 0, 0
        for shape in self.shapes:
            begin = end
            end += reduce(lambda x, y: x * y, shape)
            p.append(params[0, begin:end].reshape(shape))
        return p

    def initialize(self):
        cos_coef = torch.zeros(self.order, self.start.shape[1])
        sin_coef = torch.zeros(self.order, self.start.shape[1])
        if self.cos_coef is None or self.sin_coef is None:
            self.cos_coef = nn.Parameter(cos_coef, requires_grad=True)
            self.sin_coef = nn.Parameter(sin_coef, requires_grad=True)
        else:
            self.cos_coef.data = cos_coef.to(self.cos_coef.device)
            self.sin_coef.data = sin_coef.to(self.sin_coef.device)

    def tangent(self, t):
        sin_bias = torch.sin(self.frequency * t)
        cos_bias = torch.cos(self.frequency * t)
        start_coef = -torch.sum(torch.mul(torch.mul(self.cos_coef, self.frequency), sin_bias), dim=0, keepdim=True) - 1
        stop__coef =  torch.sum(torch.mul(torch.mul(self.sin_coef, self.frequency), cos_bias), dim=0, keepdim=True) + 1
        parameters = self.start * start_coef + self.stop * stop__coef
        parameters = self.warp(parameters)
        return parameters

    def forward(self, t):
        cos_bias = torch.cos(self.frequency * t)
        sin_bias = torch.sin(self.frequency * t)
        start_coef = torch.sum(torch.mul(self.cos_coef, cos_bias), dim=0, keepdim=True) + (1 - t)
        stop__coef = torch.sum(torch.mul(self.sin_coef, sin_bias), dim=0, keepdim=True) + t
        parameters = self.start * start_coef + self.stop * stop__coef
        parameters = self.warp(parameters)
        return parameters


class PolygonalChain(nn.Module):
    """
    polygonal chain with one bend.
    """
    def __init__(self, start, stop, lambd=0.5):
        super(PolygonalChain, self).__init__()
        for p1, p2 in zip(start, stop):
            assert p1.shape == p2.shape

        self.eps = 1e-2
        self.stop = stop
        self.start = start
        self.lambd = lambd
        self.bend_parameters = list()
        self.initialize()

    def initialize(self):
        self.bend_parameters.clear()
        for p1, p2 in zip(self.start, self.stop):
            p = (1 - self.lambd) * p1 + self.lambd * p2
            self.bend_parameters.append(nn.Parameter(p.data, requires_grad=True))

    def correct_shape(self, parameters):
        for index in range(len(parameters)):
            param, stop_ = parameters[index], self.stop[index]
            assert param.numel() == stop_.numel()
            if param.shape != stop_.shape:
                parameters[index] = param.reshape(stop_.shape)

    def tangent(self, t):
        # assert (0 <= t & t <= 1)
        parameters = []
        if t <= self.lambd:  # 0 <= t <= lambda
            factor = 1 / self.lambd
            for p1, theta in zip(self.start, self.bend_parameters):
                parameters.append(-factor * p1 + theta * factor)
        else:  # lambda <= t <= 1
            factor = 1 / (1 - self.lambd)
            for theta, p2 in zip(self.bend_parameters, self.stop):
                parameters.append(-factor * theta + p2 * factor)
        self.correct_shape(parameters)
        return parameters

    def forward(self, t):
        # assert (0 <= t & t <= 1)
        parameters = []
        if t <= self.lambd:  # 0 <= t <= lambda
            factor = t / self.lambd
            for p1, theta in zip(self.start, self.bend_parameters):
                parameters.append(p1 * (1 - factor) + theta * factor)
        else:  # lambda <= t <= 1
            factor = (t - self.lambd) / (1 - self.lambd)
            for theta, p2 in zip(self.bend_parameters, self.stop):
                parameters.append(theta * (1 - factor) + p2 * factor)
        self.correct_shape(parameters)
        return parameters


class BezierCurve(nn.Module):
    """
    bezier curve with one bend, this implementation is the same as the original definition in
    "Garipov T, Izmailov P, Podoprikhin D, et al. Loss surfaces, mode connectivity, and fast ensembling of dnns[J].
    Advances in neural information processing systems, 2018, 31."
    """
    def __init__(self, start, stop, **kwargs):
        super(BezierCurve, self).__init__()
        for p1, p2 in zip(start, stop):
            assert p1.shape == p2.shape

        self.eps = 1e-2
        self.stop = stop
        self.start = start
        self.lambd = 0.5  # TODO
        self.bend_parameters = list()
        self.initialize()

    def initialize(self):
        self.bend_parameters.clear()
        for p1, p2 in zip(self.start, self.stop):
            p = self.lambd * p1 + self.lambd * p2
            self.bend_parameters.append(nn.Parameter(p.data, requires_grad=True))

    def correct_shape(self, parameters):
        for index in range(len(parameters)):
            param, stop_ = parameters[index], self.stop[index]
            assert param.numel() == stop_.numel()
            if param.shape != stop_.shape:
                parameters[index] = param.reshape(stop_.shape)

    def tangent(self, t):
        # assert (0 <= t & t <= 1)
        parameters = []
        for p1, theta, p2 in zip(self.start, self.bend_parameters, self.stop):
            parameters.append(p1 * (-2 + 2 * t) + theta * 2 * (1 - 2 * t) + p2 * 2 * t)
        self.correct_shape(parameters)
        return parameters

    def forward(self, t):
        # assert (0 <= t & t <= 1)
        parameters = []
        for p1, theta, p2 in zip(self.start, self.bend_parameters, self.stop):
            parameters.append(p1 * (1 - t)**2 + theta * 2 * t * (1 - t) + p2 * t**2)
        self.correct_shape(parameters)
        return parameters


class SimplicialComplex(PolygonalChain):
    """
    Benton G, Maddox W, Lotfi S, et al.
    Loss surface simplexes for mode connecting volumes and fast ensembling[C]//
    International Conference on Machine Learning. PMLR, 2021: 769-779.
    """

    def __init__(self, start, stop, lambd=0.5):
        super(SimplicialComplex, self).__init__(start, stop, lambd)
        for p1, p2 in zip(start, stop):
            assert p1.shape == p2.shape

        self.eps = 1e-2
        self.stop = stop
        self.start = start
        self.lambd = lambd
        self.vertices = list()
        self.bend_parameters = list()
        self.initialize()

    def tangent(self, t):
        # TODO, meaningless
        # assert (0 <= t & t <= 1)
        parameters = []
        if t <= self.lambd:  # 0 <= t <= lambda
            factor = 1 / self.lambd
            for p1, theta in zip(self.start, self.bend_parameters):
                parameters.append(-factor * p1 + theta * factor)
        else:  # lambda <= t <= 1
            factor = 1 / (1 - self.lambd)
            for theta, p2 in zip(self.bend_parameters, self.stop):
                parameters.append(-factor * theta + p2 * factor)
        self.correct_shape(parameters)
        return parameters

    def add_vertex(self):
        tmp = []
        for p in self.bend_parameters:
            tmp.append(p.data)
        self.vertices.append(tmp)
        self.bend_parameters.clear()
        self.initialize()

    @staticmethod
    def get_shift(vertex1, vertex2):
        shift = []
        for v1, v2 in zip(vertex1, vertex2):
            shift.append((v1 - v2).reshape(1, -1))
        shift = torch.cat(shift, dim=1)
        return shift

    def complex_volume(self):
        if len(self.vertices) == 0:
            return 0
        elif len(self.vertices) == 1:
            # Heron's formula
            edge_a = torch.norm(self.get_shift(self.start, self.bend_parameters))
            edge_b = torch.norm(self.get_shift(self.bend_parameters, self.vertices[0]))
            edge_c = torch.norm(self.get_shift(self.start, self.vertices[0]))
            half_girth = (edge_a + edge_b + edge_c) / 2
            area1 = torch.sqrt(half_girth * (half_girth - edge_a) * (half_girth - edge_b) * (half_girth -edge_c))

            edge_a = torch.norm(self.get_shift(self.stop, self.bend_parameters))
            edge_b = torch.norm(self.get_shift(self.bend_parameters, self.vertices[0]))
            edge_c = torch.norm(self.get_shift(self.stop, self.vertices[0]))
            half_girth = (edge_a + edge_b + edge_c) / 2
            area2 = torch.sqrt(half_girth * (half_girth - edge_a) * (half_girth - edge_b) * (half_girth - edge_c))
            return area1 + area2
        else:
            raise NotImplementedError

    def forward(self, t):
        # assert (0 <= t & t <= 1)
        parameters = []
        if len(self.vertices) == 0:
            if t <= self.lambd:  # 0 <= t <= lambda
                factor = t / self.lambd
                for p1, theta in zip(self.start, self.bend_parameters):
                    parameters.append(p1 * (1 - factor) + theta * factor)
            else:  # lambda <= t <= 1
                factor = (t - self.lambd) / (1 - self.lambd)
                for theta, p2 in zip(self.bend_parameters, self.stop):
                    parameters.append(theta * (1 - factor) + p2 * factor)

        elif len(self.vertices) == 1:
            if t <= self.lambd:  # 0 <= t <= lambda
                factor = t / self.lambd
                ratio = np.random.uniform()
                for p1, theta, verte in zip(self.start, self.bend_parameters, self.vertices[0]):
                    parameters.append((p1 * (1 - factor) + theta * factor) * (1 - ratio) +
                                      (p1 * (1 - factor) + verte * factor) * ratio)
            else:  # lambda <= t <= 1
                factor = (t - self.lambd) / (1 - self.lambd)
                ratio = np.random.uniform()
                for theta, p2, verte in zip(self.bend_parameters, self.stop, self.vertices[0]):
                    parameters.append((theta * (1 - factor) + p2 * factor) * (1 - ratio) +
                                      (verte * (1 - factor) + p2 * factor) * ratio)

        elif len(self.vertices) == 2:
            # TODO, temporary for ensemble
            if t == self.lambd:  # 0 <= t <= lambda
                factor = t / self.lambd
                ratio = 0.5
                for p1, verte1, verte2 in zip(self.start, self.vertices[0], self.vertices[1]):
                    parameters.append((p1 * (1 - factor) + verte1 * factor) * (1 - ratio) +
                                      (p1 * (1 - factor) + verte2 * factor) * ratio)
            else:  # lambda <= t <= 1
                raise ValueError
        else:
            raise NotImplementedError
        self.correct_shape(parameters)
        return parameters


class TOrderBezierCurve(nn.Module):
    """
    bezier curve with two bends (i.e., third order), this implementation is the same as the original definition in
    "Garipov T, Izmailov P, Podoprikhin D, et al. Loss surfaces, mode connectivity, and fast ensembling of dnns[J].
    Advances in neural information processing systems, 2018, 31."
    """
    def __init__(self, start, stop, **kwargs):
        super(TOrderBezierCurve, self).__init__()
        for p1, p2 in zip(start, stop):
            assert p1.shape == p2.shape

        self.eps = 1e-2
        self.stop = stop
        self.start = start
        self.lambd1 = 1/3
        self.lambd2 = 2/3
        self.bend_parameters = []
        self.layers_num = len(self.stop)
        self.initialize()

    def initialize(self):
        self.bend_parameters.clear()
        for p1, p2 in zip(self.start, self.stop):
            p = self.lambd2 * p1 + self.lambd1 * p2
            self.bend_parameters.append(nn.Parameter(p.data, requires_grad=True))

        for p1, p2 in zip(self.start, self.stop):
            p = self.lambd1 * p1 + self.lambd2 * p2
            self.bend_parameters.append(nn.Parameter(p.data, requires_grad=True))

    def correct_shape(self, parameters):
        for index in range(len(parameters)):
            param, stop_ = parameters[index], self.stop[index]
            assert param.numel() == stop_.numel()
            if param.shape != stop_.shape:
                parameters[index] = param.reshape(stop_.shape)

    def tangent(self, t):
        # assert (0 <= t & t <= 1)
        parameters = []
        for p1, w1, w2, p2 in zip(self.start, self.bend_parameters[:self.layers_num],
                                  self.bend_parameters[self.layers_num:], self.stop):
            parameters.append(p1 * (-3 * (t - 1)**2) + w1 * (3 * (3 * t - 1) * (t - 1)) + w2 * (3 * t * (-3 * t + 2)) +
                              p2 * 3 * t ** 2)
        self.correct_shape(parameters)
        return parameters

    def forward(self, t):
        # assert (0 <= t & t <= 1)
        parameters = []
        for p1, w1, w2, p2 in zip(self.start, self.bend_parameters[:self.layers_num],
                                  self.bend_parameters[self.layers_num:], self.stop):
            parameters.append(p1 * ((1 - t) ** 3) + w1 * (3 * t * (1 - t) ** 2) + w2 * (3 * t ** 2 * (1 - t)) +
                              p2 * t ** 3)
        self.correct_shape(parameters)
        return parameters
