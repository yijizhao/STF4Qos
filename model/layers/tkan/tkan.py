import torch
import torch.nn as nn
from model.layers.tkan.spline import BSplineActivation, FixedSplineActivation, PowerSplineActivation

"""
Reference:
Genet, Remi and Inzirillo, Hugo. A Temporal Kolmogorov-Arnold Transformer for Time Series Forecasting, 2024, arXiv preprint arXiv:2406.02486.
"""
class DropoutRNNCell:
    def __init__(self, dropout: float = 0.0, recurrent_dropout: float = 0.0):
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def get_dropout_mask(self, step_input):
        if self._dropout_mask is None and self.dropout > 0:
            self._dropout_mask = torch.dropout(torch.ones_like(step_input), p=self.dropout, training=True)
        return self._dropout_mask

    def get_recurrent_dropout_mask(self, step_input):
        if self._recurrent_dropout_mask is None and self.recurrent_dropout > 0:
            self._recurrent_dropout_mask = torch.dropout(torch.ones_like(step_input), p=self.recurrent_dropout,
                                                     training=True)
        return self._recurrent_dropout_mask

    def reset_dropout_mask(self):
        self._dropout_mask = None

    def reset_recurrent_dropout_mask(self):
        self._recurrent_dropout_mask = None


class TKANCell(nn.Module, DropoutRNNCell):
    def __init__(self, units, tkan_activations=None, activation="tanh", recurrent_activation="sigmoid",
                 use_bias=True, dropout=0.0, recurrent_dropout=0.0):
        super(TKANCell, self).__init__()
        DropoutRNNCell.__init__(self, dropout, recurrent_dropout)
        self.units = units
        self.activation = getattr(torch, activation)
        self.recurrent_activation = getattr(torch, recurrent_activation)
        self.use_bias = use_bias

        self.kernel = nn.Parameter(torch.Tensor(units, 3 * units))
        self.recurrent_kernel = nn.Parameter(torch.Tensor(units, 3 * units))
        self.bias = nn.Parameter(torch.Tensor(3 * units)) if use_bias else None

        self.tkan_sub_layers = nn.ModuleList(tkan_activations or [BSplineActivation(3)])

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, states):
        h_tm1, c_tm1, *sub_states = states

        dp_mask = self.get_dropout_mask(inputs)
        rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)

        if self.training and self.dropout > 0.0:
            inputs = inputs * dp_mask
        if self.training and self.recurrent_dropout > 0.0:
            h_tm1 = h_tm1 * rec_dp_mask

        if self.use_bias:
            z = torch.matmul(inputs, self.kernel) + torch.matmul(h_tm1, self.recurrent_kernel) + self.bias
        else:
            z = torch.matmul(inputs, self.kernel) + torch.matmul(h_tm1, self.recurrent_kernel)

        i, f, o = torch.chunk(self.recurrent_activation(z), 3, dim=-1)
        c = f * c_tm1 + i * self.activation(inputs)

        h = o * self.activation(c)

        return h, [h, c]

    def get_initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.units, device=device)
        c = torch.zeros(batch_size, self.units, device=device)
        sub_states = [torch.zeros(batch_size, 1, device=device) for _ in range(len(self.tkan_sub_layers))]
        return [h, c] + sub_states


class TKAN(nn.Module):
    def __init__(self, units, tkan_activations=None, activation="tanh", recurrent_activation="sigmoid", use_bias=True,
                 dropout=0.0, recurrent_dropout=0.0):
        super(TKAN, self).__init__()
        self.cell = TKANCell(units, tkan_activations, activation, recurrent_activation, use_bias, dropout,
                             recurrent_dropout)

    def forward(self, inputs, initial_state=None):
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device
        if initial_state is None:
            initial_state = self.cell.get_initial_state(batch_size, device)

        states = initial_state
        outputs = []
        for t in range(seq_len):
            output, states = self.cell(inputs[:, t, :], states)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1), states
class GRUCell(nn.Module, DropoutRNNCell):
    def __init__(self, units, tkan_activations=None, activation="tanh", recurrent_activation="sigmoid",
                 use_bias=False, dropout=0.0, recurrent_dropout=0.0):
        super(GRUCell, self).__init__()
        DropoutRNNCell.__init__(self, dropout, recurrent_dropout)
        self.units = units
        self.activation = getattr(torch, activation)
        self.recurrent_activation = getattr(torch, recurrent_activation)
        self.use_bias = use_bias

        self.kernel = nn.Parameter(torch.Tensor(units, 2 * units))
        self.recurrent_kernel = nn.Parameter(torch.Tensor(units, 2 * units))
        self.bias = nn.Parameter(torch.Tensor(2 * units)) if use_bias else None

        self.kernel1 = nn.Parameter(torch.Tensor(units, units))
        self.recurrent_kernel1 = nn.Parameter(torch.Tensor(units, units))
        self.tkan_sub_layers = nn.ModuleList(tkan_activations or [BSplineActivation(3)])

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, states):
        h_tm1, c_tm1, *sub_states = states

        dp_mask = self.get_dropout_mask(inputs)
        rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)

        if self.training and self.dropout > 0.0:
            inputs = inputs * dp_mask
        if self.training and self.recurrent_dropout > 0.0:
            h_tm1 = h_tm1 * rec_dp_mask

        if self.use_bias:
            e = torch.matmul(inputs, self.kernel) + torch.matmul(h_tm1, self.recurrent_kernel) + self.bias
        else:
            e = torch.matmul(inputs, self.kernel) + torch.matmul(h_tm1, self.recurrent_kernel)

        z, r = torch.chunk(self.recurrent_activation(e), 2, dim=-1)

        h_ = self.activation(torch.matmul(inputs, self.kernel1) + r * torch.matmul(h_tm1, self.recurrent_kernel1))
        h = z * h_tm1 + (1 - z) * h_

        return h, [h]

    def get_initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.units, device=device)
        sub_states = [torch.zeros(batch_size, 1, device=device) for _ in range(len(self.tkan_sub_layers))]
        return [h] + sub_states