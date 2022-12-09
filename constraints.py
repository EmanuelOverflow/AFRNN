import torch


class Constraint(object):
    def __call__(self, module):
        if hasattr(module, 'kernel_constraint'):
            if hasattr(module, 'weight'):
                if module.kernel_constraint is not None:
                    self.call(module)

    def call(self, module):
        pass


class MinMaxWeightConstraint(Constraint):

    def __init__(self, min_value=0., max_value=1., dim=0):
        self.min_value = min_value
        self.max_value = max_value
        self.dim = dim

    def call(self, module):
        # filter the variables to get the ones you want
        with torch.no_grad():
            w = module.weight
            w_min = w.min(dim=self.dim)[0]
            w_max = w.max(dim=self.dim)[0]
            w = (w - w_min) / (w_max - w_min)
            w = w * (self.max_value - self.min_value) + self.min_value
            module.weight.data.copy_(w)


class ClipWeightValueConstraint(Constraint):

    def __init__(self, min_value=0., max_value=1., dim=0):
        self.min_value = min_value
        self.max_value = max_value
        self.dim = dim

    def call(self, module):
        with torch.no_grad():
            w = module.weight
            module.weight.data.copy_(torch.where(w < 1e-6, torch.ones_like(w) * self.min_value,
                                                 torch.where(w > 1., torch.ones_like(w) * self.max_value, w)))


class SigmoidConstraint(Constraint):

    def call(self, module):
        with torch.no_grad():
            w = module.weight
            module.weight.data.copy_(torch.sigmoid(w))


class GaussianConstraint(Constraint):

    def call(self, module):
        with torch.no_grad():
            w = module.weight
            module.weight.data.copy_(torch.exp(-(w ** 2) / 2))


class Sinc2Constraint(Constraint):

    pi = torch.tensor(3.141592653589793)

    def _sinc(self, x):
        sinc = lambda inp: torch.sin(self.pi * inp) / (self.pi * inp)
        return torch.where(x > 0, sinc(x)**2, torch.ones_like(x))

    def call(self, module):
        with torch.no_grad():
            w = module.weight
            module.weight.data.copy_(self._sinc(w))

