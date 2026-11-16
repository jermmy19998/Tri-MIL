import torch
import torch.nn as nn
import numpy as np
from .functional import Top1_Hard_SVM, Topk_Hard_SVM, Top1_Smooth_SVM, Topk_Smooth_SVM

from .utils import detect_large


class _SVMLoss(nn.Module):
    def __init__(self, n_classes, alpha):
        """Initialize the _SVMLoss instance and configure its dependencies."""
        assert isinstance(n_classes, int)

        assert n_classes > 0
        assert alpha is None or alpha >= 0

        super(_SVMLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 1
        self.register_buffer("labels", torch.from_numpy(np.arange(n_classes)))
        self.n_classes = n_classes
        self._tau = None

    def forward(self, x, y):
        """Compute the _SVMLoss forward pass for the supplied input tensors."""
        raise NotImplementedError("Forward needs to be re-implemented for each loss")

    @property
    def tau(self):
        """Tau."""
        return self._tau

    @tau.setter
    def tau(self, tau):
        """Tau."""
        if self._tau != tau:
            # print("Setting tau to {}".format(tau))
            self._tau = float(tau)
            self.get_losses()

    def cuda(self, device=None):
        """Cuda."""
        nn.Module.cuda(self, device)
        self.get_losses()
        return self

    def cpu(self):
        """Cpu."""
        nn.Module.cpu()
        self.get_losses()
        return self


class MaxTop1SVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None):
        """Initialize the MaxTop1SVM instance and configure its dependencies."""
        super(MaxTop1SVM, self).__init__(n_classes=n_classes, alpha=alpha)
        self.get_losses()

    def forward(self, x, y):
        """Compute the MaxTop1SVM forward pass for the supplied input tensors."""
        return self.F(x, y).mean()

    def get_losses(self):
        """Get losses."""
        self.F = Top1_Hard_SVM(self.labels, self.alpha)


class MaxTopkSVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None, k=5):
        """Initialize the MaxTopkSVM instance and configure its dependencies."""
        super(MaxTopkSVM, self).__init__(n_classes=n_classes, alpha=alpha)
        self.k = k
        self.get_losses()

    def forward(self, x, y):
        """Compute the MaxTopkSVM forward pass for the supplied input tensors."""
        return self.F(x, y).mean()

    def get_losses(self):
        """Get losses."""
        self.F = Topk_Hard_SVM(self.labels, self.k, self.alpha)


class SmoothTop1SVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None, tau=1.0):
        """Initialize the SmoothTop1SVM instance and configure its dependencies."""
        super(SmoothTop1SVM, self).__init__(n_classes=n_classes, alpha=alpha)
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y):
        """Compute the SmoothTop1SVM forward pass for the supplied input tensors."""
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
        """Get losses."""
        self.F_h = Top1_Hard_SVM(self.labels, self.alpha)
        self.F_s = Top1_Smooth_SVM(self.labels, self.tau, self.alpha)


class SmoothTopkSVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None, tau=1.0, k=5):
        """Initialize the SmoothTopkSVM instance and configure its dependencies."""
        super(SmoothTopkSVM, self).__init__(n_classes=n_classes, alpha=alpha)
        self.k = k
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y):
        """Compute the SmoothTopkSVM forward pass for the supplied input tensors."""
        smooth, hard = detect_large(x, self.k, self.tau, self.thresh)

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
        """Get losses."""
        self.F_h = Topk_Hard_SVM(self.labels, self.k, self.alpha)
        self.F_s = Topk_Smooth_SVM(self.labels, self.k, self.tau, self.alpha)
