import torch
from torch import nn

from model.layers.fmlayer import MultiLayerPerceptron
from model.layers.tkan.spline import BSplineActivation, FixedSplineActivation
from model.layers.tkan.tkan import TKAN


class NeuralTensorFactorization(nn.Module):
    """
    Neural Tensor Factorization (NTF)

    Reference:
    Xian Wu, et al. Neural Tensor Factorization for Temporal Interaction Learning. WSDM 2019.

    Core idea:
    - Learn embeddings for user, item, and time
    - Model temporal dynamics using sequence modeling (LSTM / GRU / TKAN)
    - Capture high-order interactions via MLP (neural tensor factorization)
    """

    def __init__(self, field_dims, field_order, config):
        super().__init__()

        # Field configuration
        self.field_order = field_order
        self.num_times = field_dims[0]
        self.num_users = field_dims[1]
        self.num_items = field_dims[2]

        # Hyperparameters
        self.embed_dim = config['embed_dim']
        self.mlp_dims = config['NTF']['mlp_dims']
        self.dropout = config['NTF']['dropout']
        self.num_layers = config['NTF']['num_layers']
        self.step = config['NTF']['step']   # length of temporal sequence
        self.device = config['device']

        # =====================================================
        # Embedding initialization
        # Node space includes:
        # - users
        # - items
        # - time slices (historical + current)
        # =====================================================
        num_nodes = self.num_times + self.num_users + self.num_items + self.step

        self.embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(num_nodes, self.embed_dim))
        )

        # MLP for nonlinear interaction modeling (user-item-time)
        self.mlp = MultiLayerPerceptron(
            self.embed_dim * 3,
            self.mlp_dims,
            self.dropout
        ).to(self.device)

        # =====================================================
        # Temporal modeling module (sequence encoder)
        # Supports: LSTM / GRU / TKAN
        # =====================================================
        self.rnn_mode = 'tkan'

        if self.rnn_mode == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=self.embed_dim,
                num_layers=self.num_layers,
                batch_first=True
            )

        elif self.rnn_mode == 'gru':
            self.rnn = nn.GRU(
                input_size=self.embed_dim,
                hidden_size=self.embed_dim,
                num_layers=self.num_layers,
                batch_first=True
            )

        elif self.rnn_mode == 'tkan':
            # TKAN: Temporal Kernel Activation Network
            # Uses spline-based activation functions for flexible temporal modeling
            self.tkan = TKAN(
                units=self.embed_dim,
                tkan_activations=[
                    BSplineActivation(3),
                    FixedSplineActivation(2.0)
                ]
            )

        # Projection layer to refine temporal embedding
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

        # Final prediction layer
        self.fc = nn.Linear(self.mlp_dims[-1], 1)

        print(f"rnn_mode: {self.rnn_mode}")

    def forward(self, x):
        """
        Forward pass of Neural Tensor Factorization.

        Args:
            x: Long tensor of shape (batch_size, 3)
               [time_id, user_id, item_id]

        Returns:
            Predicted QoS values
        """

        t_id, u_id, i_id = x[:, 0], x[:, 1], x[:, 2]

        # =====================================================
        # Embedding lookup
        # =====================================================

        # User embedding
        u_emb = self.embedding[u_id]

        # Item embedding (offset by num_users)
        i_emb = self.embedding[i_id + self.num_users]

        # =====================================================
        # Construct temporal sequence (historical time slices)
        # =====================================================

        input_slice_ids = list()

        # Offset for time indices in embedding table
        t_id = self.num_users + self.num_items + self.step + t_id

        # Collect previous "step" time slices (sliding window)
        for s in range(self.step, 0, -1):
            input_slice_ids.append(t_id - s)

        # Shape: (batch_size, step)
        input_slice_ids = torch.stack(input_slice_ids, dim=1)

        # Time sequence embeddings: (batch_size, step, embed_dim)
        t_emb = self.embedding[input_slice_ids]

        # =====================================================
        # Temporal sequence modeling
        # =====================================================

        if self.rnn_mode == 'lstm':
            # Initialize hidden and cell states
            c0 = torch.zeros(self.num_layers, u_id.shape[0], self.embed_dim).to(self.device)
            h0 = torch.zeros(self.num_layers, u_id.shape[0], self.embed_dim).to(self.device)

            # LSTM forward propagation
            output, (hs, cs) = self.rnn(t_emb, (h0, c0))

            # Use final hidden state as temporal representation
            t_emb = torch.sigmoid(self.projection(hs.squeeze(0)))

        if self.rnn_mode == 'gru':
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, u_id.shape[0], self.embed_dim).to(self.device)

            # GRU forward propagation
            output, (hs) = self.rnn(t_emb, (h0))

            t_emb = torch.sigmoid(self.projection(hs.squeeze(0)))

        if self.rnn_mode == 'tkan':
            # Initialize TKAN hidden state
            h0 = self.tkan.cell.get_initial_state(u_id.shape[0], self.device)

            # TKAN forward propagation
            _, hs = self.tkan(t_emb, h0)

            # Use final hidden state
            t_emb = torch.sigmoid(self.projection(hs[0]))

        # =====================================================
        # Neural tensor interaction
        # =====================================================

        # Concatenate user, item, and temporal embeddings
        z = self.mlp(torch.cat([u_emb, i_emb, t_emb], dim=1))

        # Final prediction
        y = self.fc(z).squeeze(-1)

        return y