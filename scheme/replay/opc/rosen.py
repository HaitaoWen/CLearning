import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required


class Rosen(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, boundary=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Rosen, self).__init__(params, defaults)

        self.eps = 1e-5
        order = params[0].shape[1]
        device = params[0].device
        A1 = torch.eye(order)
        A2 = 0. - torch.eye(order).float()
        self.order = order
        self.device = device
        self.lr = torch.tensor(lr).to(device)
        # Inequality constraint coefficient (normal)
        self.A = torch.cat([A1, A2], dim=0).to(device)
        # Equality constraint coefficient
        self.E = torch.ones(1, order).float().to(device)
        # Inequality boundary
        if boundary is None:
            self.b = None
        else:
            self.b = (0. - torch.ones(order * 2, 1) * boundary).to(device)
        # Equality constraint
        self.e = torch.tensor(0.).to(device)

    def __setstate__(self, state):
        super(Rosen, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def correct(self, D, M, underdetermined):
        # Make up for errors caused by inversion
        # This scheme can never get an exact theoretical solution.
        # mask = D.abs() < 1e-4
        # D.masked_fill_(mask, 0)
        # mean = D[~mask].mean(dim=0)
        # D[~mask] = D[~mask] - mean
        # We try to add a small perturbation in the normal direction to D
        if underdetermined:
            norm = M[:-1, :]
        else:
            norm = M
        D = F.normalize(D, p=2, dim=0)
        if norm.numel() > 0:
            index = norm.nonzero()[:, 1]
            sign = norm.sum(dim=1).sign().unsqueeze(dim=1)

            # P = D[index] * sign
            # mask = (P >= self.eps).squeeze(dim=1)
            # if not mask.all():
            #     mask = ~mask
            #     perturbation = torch.zeros_like(D)
            #     perturbation[index[mask]] = sign[mask] * (P[mask].abs() + 1e-2)
            #     D = D + perturbation

            perturbation = torch.zeros_like(D)
            perturbation[index] = sign * D.abs().max(dim=0)[0] / 20
            D = D + perturbation
        mean = D.mean(dim=0)
        D = D - mean
        return D

    @staticmethod
    def coordinate(M, G):
        W = (M @ M.T).inverse() @ M @ G
        return W

    @staticmethod
    def projection(G, M, W):
        D = -G + M.T @ W
        return D

    def maximum_step(self, A2, b2, x, d):
        index = A2.nonzero()[:, 1]
        sign = A2.sum(dim=1).sign().unsqueeze(dim=1)
        b_hat = b2 - x[index] * sign  # b_hat = b2 - A2 @ x
        d_hat = d[index] * sign  # d_hat = A2 @ d
        mask = d_hat >= 0
        if mask.all():
            step = self.lr
        else:
            mask_ = b_hat < 0
            mask = (~mask) & mask_
            max_step = (b_hat[mask] / d_hat[mask]).min()
            max_step = (max_step / 1e-3).floor() * 1e-3
            step = torch.minimum(max_step, self.lr)
            assert step >= 0, '{} \n {} \n{} step must > 0'.format(b_hat[mask].cpu().numpy(),
                                                                   d_hat[mask].cpu().numpy(), step)
        return step

    def feasible_direction(self, A1, A2, b2, x, g):

        underdetermined = True
        if A1.shape[0] == self.order:  # and ((A1.abs().sum(dim=0) - self.E).abs() < self.eps).all():
            # fast check whether M is full row rank
            M = A1
            underdetermined = False
        else:
            M = torch.cat([A1, self.E], dim=0)

        if M.numel() == 0:
            # if M is empty, stop projection,
            # we have equality constraint, so M never gets empty.
            skip = False
            step = self.maximum_step(A2, b2, x, -g)
            return -g, step, skip
        W = self.coordinate(M, g)
        d = self.projection(g, M, W)
        d = self.correct(d, M, underdetermined)
        zero_flag = d.abs().sum() < self.eps
        if zero_flag and W.shape[0] > 1:
            if underdetermined:
                W = W[:-1]
            if (W >= self.eps).all():
                # for current data, x is a KT-point,
                # stop computation
                skip = True
                return -1, -1, skip
            else:
                # if W contains negative component, remove one.
                _, index = torch.min(W, dim=0)
                mask = torch.arange(A1.shape[0]).to(self.device) != index
                A1 = A1[mask]
                d, step, skip = self.feasible_direction(A1, A2, b2, x, g)
                return d, step, skip
        else:
            # current direction is feasible,
            # solve the maximum step
            step = self.maximum_step(A2, b2, x, d)
            skip = False
            return d, step, skip

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            # momentum = group['momentum']
            # dampening = group['dampening']
            # nesterov = group['nesterov']

            for X in group['params']:
                if X.grad is None:
                    continue
                G = X.grad.data

                # ****************** feasible direction ****************** #
                X, G = X.unsqueeze(dim=2), G.unsqueeze(dim=2)
                for index, (x, g) in enumerate(zip(X, G)):
                    if weight_decay != 0:
                        g.add_(x.data, alpha=weight_decay)

                    if self.b is None:
                        W = self.coordinate(self.E, g)
                        d = self.projection(g, self.E, W)
                        mean = d.mean(dim=0)
                        d = d - mean
                        x.data.add_(d, alpha=self.lr)

                        # check constraint
                        # assert (x.sum(dim=0) - self.e).abs() < self.eps, 'Equality constraint out of bounds!, {}' \
                        #     .format((x.sum(dim=0) - self.e).abs().cpu().data)
                    else:
                        b_ = torch.cat([x, -x], dim=0)  # b_ = self.A @ x
                        c_ = torch.cat([-g, g], dim=0) / torch.norm(g)  # c_ = self.A @ (-g) / torch.norm(g)
                        mask_ = (c_ < self.eps).squeeze(dim=1)
                        mask = ((b_ - self.b).abs() < self.eps).squeeze(dim=1)
                        mask = mask & mask_
                        A1, b1 = self.A[mask], self.b[mask]
                        A2, b2 = self.A[~mask], self.b[~mask]
                        d, step, skip = self.feasible_direction(A1, A2, b2, x, g)
                        if skip:
                            continue
                        x.data.add_(d, alpha=step)

                        # check constraint
                        assert (x.sum(dim=0) - self.e).abs() < self.eps, 'Equality constraint out of bounds!, {}' \
                            .format((x.sum(dim=0) - self.e).abs().cpu().data)
                        # b_ = torch.cat([x, -x], dim=0)
                        # assert (b_ >= self.b - self.eps).all(), 'Inequality constraint out of bounds!, ' \
                        #                                         'b_ {}, mask {}, d {} step {}, index {} A1 {}' \
                        #     .format(b_.cpu().data.numpy(), mask.cpu().data, d.cpu().data.numpy(),
                        #             step.cpu().data.numpy(), index, A1)
        return loss
