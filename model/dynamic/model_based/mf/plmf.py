import torch
from torch import nn
from torch.autograd import Variable


import torch
from torch import nn

class PersonalizedLSTM(nn.Module):
    """Personalized LSTM Cell"""
    def __init__(self, num, K):
        super().__init__()
        self.K = K
        # Initialize gates and state
        self.gates = nn.ModuleDict({
            'input': nn.ModuleDict({'W': nn.Embedding(num, K), 'U': nn.Linear(K, K)}),
            'forget': nn.ModuleDict({'W': nn.Embedding(num, K), 'U': nn.Linear(K, K)}),
            'output': nn.ModuleDict({'W': nn.Embedding(num, K), 'U': nn.Linear(K, K)}),
            'state': nn.ModuleDict({'W': nn.Embedding(num, K), 'U': nn.Linear(K, K)})
        })
        # Initialize weights
        for gate in self.gates.values():
            nn.init.orthogonal_(gate['W'].weight)
            nn.init.orthogonal_(gate['U'].weight)
        nn.init.constant_(self.gates['forget']['U'].bias, 1.0)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, X, h, C):
        forget_gate = torch.sigmoid(self.gates['forget']['W'](X) + self.gates['forget']['U'](h))
        input_gate = torch.sigmoid(self.gates['input']['W'](X) + self.gates['input']['U'](h))
        new_info = torch.tanh(self.gates['state']['W'](X) + self.gates['state']['U'](h))
        new_C = forget_gate * C + self.dropout(input_gate) * new_info
        output_gate = torch.sigmoid(self.gates['output']['W'](X) + self.gates['output']['U'](h))
        new_h = self.dropout(output_gate) * torch.tanh(new_C)
        return new_h, new_C

class PersonalizedLSTMBasedMatrixFactorization(nn.Module):
    """ Personalized LSTM Based Matrix Factorization for Online QoS Prediction.ICWS2017."""
    def __init__(self, field_dims, field_order, config):
        super().__init__()
        self.num_users, self.num_items = field_dims[1], field_dims[2]
        self.embed_dim = config['embed_dim']
        self.step = config['PLMF']['step']
        self.device = config['device']
        # Initialize user and item LSTM cells
        self.user_cell = PersonalizedLSTM(self.num_users, self.embed_dim)
        self.item_cell = PersonalizedLSTM(self.num_items, self.embed_dim)
        # Initialize hidden and cell states
        self.user_h = torch.zeros(self.num_users, self.embed_dim, device=self.device)
        self.user_C = torch.zeros(self.num_users, self.embed_dim, device=self.device)
        self.item_h = torch.zeros(self.num_items, self.embed_dim, device=self.device)
        self.item_C = torch.zeros(self.num_items, self.embed_dim, device=self.device)

    def forward(self, x):
        t_id, u_id, i_id = x[:, 0], x[:, 1], x[:, 2]
        users_h, users_C = self.user_h[u_id], self.user_C[u_id]
        items_h, items_C = self.item_h[i_id], self.item_C[i_id]
        for _ in range(self.step):
            users_h, users_C = self.user_cell(u_id, users_h, users_C)
            items_h, items_C = self.item_cell(i_id, items_h, items_C)
        return torch.sum(users_h * items_h, dim=1)