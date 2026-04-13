import torch
from torch import nn
from torch import Tensor


class MultiLayerPerceptron(nn.Module):
    """
    Standard Multi-Layer Perceptron (MLP)

    Architecture:
        Linear -> BatchNorm -> ReLU -> Dropout

    """

    def __init__(self, input_dim: int, embed_dims: list, dropout: float,
                 output_layer=False, output_bias=False):
        super().__init__()

        layers = []

        # Build hidden layers
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))      # Fully connected layer
            layers.append(nn.BatchNorm1d(embed_dim))            # Batch normalization
            layers.append(nn.ReLU())                            # Activation function
            layers.append(nn.Dropout(p=dropout))                # Dropout for regularization
            input_dim = embed_dim

        # Optional output layer
        if output_layer:
            layers.append(nn.Linear(input_dim, 1, bias=output_bias))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        Forward pass

        :param x: Tensor of shape (batch_size, input_dim)
        """
        return self.mlp(x)


class RecurrentNeuralCollaborativeFiltering(nn.Module):
    """
    Recurrent Neural Collaborative Filtering (RNCF)

    Reference:
    RNN-based collaborative filtering for QoS prediction.

    Key ideas:
    - Each time slice has independent user/item embeddings
    - Construct temporal sequences within a time window
    - Use GRU to model temporal dynamics
    - Use MLP for nonlinear interaction modeling
    """

    def __init__(self, field_dims, field_order, config):
        super().__init__()

        # Basic configurations
        self.field_order = field_order
        self.num_times = field_dims[0]
        self.num_users = field_dims[1]
        self.num_items = field_dims[2]

        self.embed_dim = config['embed_dim']
        self.mlp_dims = config['RNCF']['mlp_dims']
        dropout = config['RNCF']['dropout']

        self.gru_layers = config['RNCF']['gru_layers']
        self.time_window = config['RNCF']['time_window']
        self.device = config['device']

        # ======================================
        # Time-aware embeddings
        # Each time slice has independent embeddings
        # ======================================
        self.user_embedding = [None] * (self.num_times + self.time_window)
        self.item_embedding = [None] * (self.num_times + self.time_window)

        for t in range(self.num_times + self.time_window):
            # User embedding at time t
            self.user_embedding[t] = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(self.num_users, self.embed_dim)
                )
            ).to(self.device)

            # Item embedding at time t
            self.item_embedding[t] = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(self.num_items, self.embed_dim)
                )
            ).to(self.device)

        # ======================================
        # MLP (kept unchanged)
        # ======================================
        self.mlp = MultiLayerPerceptron(
            self.embed_dim * 2,
            self.mlp_dims,
            dropout
        ).to(self.device)

        # ======================================
        # GRU for temporal modeling
        # ======================================
        self.rnn = nn.GRU(
            input_size=self.embed_dim * 2,
            hidden_size=self.embed_dim * 2,
            num_layers=self.gru_layers,
            batch_first=True
        )

        # Final prediction layer
        self.fc = nn.Linear(self.mlp_dims[-1], 1, bias=False)

    
    def safe_mlp_forward(self, x):
        """
        Handle the BatchNorm issue when batch size = 1.

        During training:
        - BatchNorm requires batch size > 1
        - If batch size == 1, switch to eval mode temporarily

        :param x: Tensor of shape (batch_size, dim)
        """

        if x.shape[0] == 1 and self.training:
            # Temporarily switch to evaluation mode
            self.mlp.eval()
            out = self.mlp(x)
            # Restore training mode
            self.mlp.train()
            return out

        return self.mlp(x)

    def predict(self, _user_id, _item_id, slice_id_in_window):
        """
        Predict QoS values within a time window

        :param _user_id: Tensor of user indices (sub-batch)
        :param _item_id: Tensor of item indices (sub-batch)
        :param slice_id_in_window: list of time indices
        """

        input_seq = []

        # Construct temporal input sequence
        for s in slice_id_in_window:
            x_u = self.user_embedding[s][_user_id]  # (batch, embed_dim)
            x_i = self.item_embedding[s][_item_id]  # (batch, embed_dim)

            # Concatenate user and item embeddings
            input_seq.append(torch.cat([x_u, x_i], dim=-1))

        # Stack into sequence tensor
        input_seq = torch.stack(input_seq, dim=1)
        # Shape: (batch, time_window+1, embed_dim*2)

        # Initialize hidden state
        h0 = torch.zeros(
            self.gru_layers,
            _user_id.shape[0],
            self.embed_dim * 2
        ).to(self.device)

        # GRU forward
        _, h = self.rnn(input_seq, h0)

        # Use the last hidden state
        h_last = h.squeeze(0)  # (batch, embed_dim*2)

        
        feat_t = self.safe_mlp_forward(h_last)

        # Final prediction
        return self.fc(feat_t).squeeze(-1)

    def forward(self, x):
        """
        Forward pass

        :param x: Tensor of shape (batch_size, 3)
                  [time_id, user_id, item_id]
        """

        time_id = x[:, 0]
        user_id = x[:, 1]
        item_id = x[:, 2]

        # Initialize output
        y = torch.zeros(x.shape[0]).to(self.device)

        # ======================================
        # Group samples by time slice
        # This may create sub-batches of size 1
        # ======================================
        for t in set(time_id.cpu().numpy().tolist()):

            # Indices of samples with time t
            ind = torch.nonzero(torch.eq(time_id, t)).squeeze()

            # Handle single-element case
            if ind.ndim == 0:
                ind = ind.unsqueeze(-1)

            # Sub-batch prediction
            sub_y = self.predict(
                user_id[ind],
                item_id[ind],
                list(range(t, t + self.time_window + 1))
            )

            # Write back results
            y.index_put_((ind,), sub_y)

        return y