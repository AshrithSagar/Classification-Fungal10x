"""
SmoothTop1SVM-torch.py
"""

import math

import numpy as np
import torch
import torch.autograd as ag
import torch.nn as nn


def Top1_Hard_SVM(labels, alpha=1.0):
    def fun(x, y):
        # max oracle
        max_, _ = (x + delta(y, labels, alpha)).max(1)
        # subtract ground truth
        loss = max_ - x.gather(1, y[:, None]).squeeze()
        return loss

    return fun


def Top1_Smooth_SVM(labels, tau, alpha=1.0):
    def fun(x, y):
        # add loss term and subtract ground truth score
        x = x + delta(y, labels, alpha) - x.gather(1, y[:, None])
        # compute loss
        loss = tau * log_sum_exp(x / tau)

        return loss

    return fun


def delta(y, labels, alpha=None):
    """
    Compute zero-one loss matrix for a vector of ground truth y
    """

    if isinstance(y, ag.Variable):
        labels = ag.Variable(labels, requires_grad=False)

    delta = torch.ne(y[:, None], labels[None, :]).float()

    if alpha is not None:
        delta = alpha * delta
    return delta


def log_sum_exp(x):
    """
    Compute log(sum(exp(x), 1)) in a numerically stable way.
    Assumes x is 2d.
    """
    max_score, _ = x.max(1)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score[:, None]), 1))


def detect_large(x, k, tau, thresh):
    top, _ = x.topk(k + 1, 1)
    # switch to hard top-k if (k+1)-largest element is much smaller
    # than k-largest element
    hard = torch.ge(top[:, k - 1] - top[:, k], k * tau * math.log(thresh)).detach()
    smooth = hard.eq(0)
    return smooth, hard


class _SVMLoss(nn.Module):

    def __init__(self, n_classes, alpha):

        assert isinstance(n_classes, int)

        assert n_classes > 0
        assert alpha is None or alpha >= 0

        super(_SVMLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 1
        self.register_buffer("labels", torch.from_numpy(np.arange(n_classes)))
        self.n_classes = n_classes
        self._tau = None

    def forward(self, x, y):

        raise NotImplementedError("Forward needs to be re-implemented for each loss")

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self._tau != tau:
            print("Setting tau to {}".format(tau))
            self._tau = float(tau)
            self.get_losses()

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.get_losses()
        return self

    def cpu(self):
        nn.Module.cpu()
        self.get_losses()
        return self


class SmoothTop1SVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None, tau=1.0):
        super(SmoothTop1SVM, self).__init__(n_classes=n_classes, alpha=alpha)
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y):
        smooth, hard = detect_large(x, 1, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.F_s(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.F_h(x_h, y_h).sum() / x.size(0)

        return loss

    def get_losses(self):
        self.F_h = Top1_Hard_SVM(self.labels, self.alpha)
        self.F_s = Top1_Smooth_SVM(self.labels, self.tau, self.alpha)
