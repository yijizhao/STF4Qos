import torch
import torch.nn as nn

"""
Reference:
Genet, Remi and Inzirillo, Hugo. A Temporal Kolmogorov-Arnold Transformer for Time Series Forecasting, 2024, arXiv preprint arXiv:2406.02486.
"""
class FixedSplineActivation(nn.Module):
    def __init__(self, exponent: float = 1.0, max_exponent: float = 9.0):
        super(FixedSplineActivation, self).__init__()
        self.exponent = torch.tensor(exponent, dtype=torch.float32)
        self.max_exponent = torch.tensor(max_exponent, dtype=torch.float32)
        self.epsilon = torch.finfo(torch.float32).eps
        self.exponent = torch.clamp(self.exponent, -self.max_exponent, self.max_exponent)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_safe = torch.clamp(inputs, self.epsilon, 1.0)
        return torch.pow(inputs_safe, self.exponent)

    def extra_repr(self) -> str:
        return f'exponent={{self.exponent}}, max_exponent={{self.max_exponent}}'


class PowerSplineActivation(nn.Module):
    def __init__(self, initial_exponent: float = 1.0, epsilon: float = 1e-7, max_exponent: float = 9.0,
                 trainable: bool = True):
        super(PowerSplineActivation, self).__init__()
        self.epsilon = epsilon
        self.max_exponent = torch.tensor(max_exponent, dtype=torch.float32)
        self.trainable = trainable

        self.exponent = nn.Parameter(torch.tensor(initial_exponent, dtype=torch.float32), requires_grad=trainable)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=trainable)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        clipped_exponent = torch.clamp(self.exponent, -self.max_exponent, self.max_exponent)
        x_safe = torch.clamp(inputs + self.bias, self.epsilon, 1.0)
        return torch.pow(x_safe, clipped_exponent)

    def extra_repr(self) -> str:
        return f'initial_exponent={{self.exponent}}, epsilon={{self.epsilon}}, max_exponent={{self.max_exponent}}, trainable={{self.trainable}}'


class LinspaceInitializer:
    def __init__(self, start: float, stop: float, num: int):
        self.start = start
        self.stop = stop
        self.num = num

    def __call__(self, shape):
        return torch.linspace(self.start, self.stop, self.num).reshape(shape)

    def __repr__(self):
        return f'LinspaceInitializer(start={{self.start}}, stop={{self.stop}}, num={{self.num}})'


class BSplineActivation(nn.Module):
    def __init__(self, num_bases: int = 10, order: int = 3):
        super(BSplineActivation, self).__init__()
        self.num_bases = num_bases
        self.order = order
        self.w = nn.Parameter(torch.empty(1))
        self.coefficients = nn.Parameter(torch.zeros(num_bases))
        self.bases = nn.Parameter(LinspaceInitializer(0.0, 1.0, num_bases)([num_bases]), requires_grad=False)

        nn.init.xavier_uniform_(torch.unsqueeze(self.w, 0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        silu = inputs * torch.sigmoid(inputs)
        spline_output = self.compute_spline(inputs)
        return self.w * (silu + spline_output)

    def compute_spline(self, x: torch.Tensor) -> torch.Tensor:
        safe_x = torch.clamp(x, torch.finfo(x.dtype).eps, 1.0)
        expanded_x = safe_x.unsqueeze(-1)
        basis_function_values = torch.pow(expanded_x - self.bases, self.order)
        spline_values = torch.sum(self.coefficients * basis_function_values, dim=-1)
        return spline_values

    def extra_repr(self) -> str:
        return f'num_bases={{self.num_bases}}, order={{self.order}}'