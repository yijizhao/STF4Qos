"""
Factorization Machine models in PyTorch,
see: https://github.com/rixwew/pytorch-fm
@author rixwew
@author Hao Wu
"""

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Linear, Conv1d


class FeaturesLinear(nn.Module):

    def __init__(self, field_dims: list, output_dim=1):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int_)

    def forward(self, x: Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims: list, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int_)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x, offset_enable=True):
        """
        :param offset_enable: using offset in default case
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if offset_enable:
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims: list, embed_dim: int):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int_)
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x: Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class FactorizationMachine(nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim: int, embed_dims: list, dropout: float, output_layer=False, output_bias=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1, bias=output_bias))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(nn.Module):

    def __init__(self, num_fields: int, embed_dim: int, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = nn.Parameter(torch.zeros(kernel_shape))
        nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x: Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class CrossNetwork(nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class CompressedInteractionNetwork(nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                           stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = Linear(fc_input_dim, 1)

    def forward(self, x: Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = torch.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))

class USFeatureFusion(nn.Module):
    def __init__(self, in_channels, conv_out_channels, embedding_dim, kernel_size, padding):
        super(USFeatureFusion, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, conv_out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(conv_out_channels, embedding_dim)
        )

    def forward(self, x):
        return self.model(x)



class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x, m):  # x: [B, F], m: [B, F]∈{0,1}
        xm = x * m
        denom = torch.clamp(m.sum(dim=-1, keepdim=True), min=1.0)
        out = (xm @ self.weight.t()) / denom
        if self.bias is not None:
            out = out + self.bias
        return out

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x, m):  # x: [B, F], m: [B, F]∈{0,1}
        xm = x * m
        denom = torch.clamp(m.sum(dim=-1, keepdim=True), min=1.0)
        out = (xm @ self.weight.t()) / denom
        if self.bias is not None:
            out = out + self.bias
        return out