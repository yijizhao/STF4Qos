import numpy as np
import torch
from scipy.sparse import dok_matrix
from torch import nn
from tqdm import tqdm

from model.layers.fmlayer import MultiLayerPerceptron
from util.sparseop import sparse_dropout, scipy_sparse_mat_to_torch_sparse_tensor


class GraphModeling(nn.Module):
    '''
    GraphModeling for Time-Aware QoS Prediction

    Reference:
    Wu et al. Effective Graph Modeling and Contrastive Learning for Time-Aware QoS Prediction,
    IEEE Transactions on Services Computing, 2024.

    Core idea:
    - Construct a heterogeneous temporal graph including time, user, and item nodes
    - Perform graph convolution (message passing) over multiple layers
    - Combine GNN embeddings with tensor interaction and MLP for final prediction
    '''

    def __init__(self, train_data, field_dims, field_order, config):
        super(GraphModeling, self).__init__()

        # Field index mapping
        self.time_fi = field_order[0][0]
        self.user_fi = field_order[1][0]
        self.item_fi = field_order[2][0]

        # Dimensions
        self.num_times = field_dims[0]
        self.num_users = field_dims[1]
        self.num_items = field_dims[2]

        # Hyperparameters
        self.k = config['GM']['k']                  # number of GNN propagation layers
        self.tao = config['GM']['tao']              # temperature (used in contrastive learning, if applicable)
        self.dropout = config['GM']['dropout']      # edge dropout for sparse graph
        self.embed_dim = config['embed_dim']
        self.device = config['device']
        self.mlp_dims = config['GM']['mlp_dims']

        print("--Construct graph and adjacent matrix...")

        # =====================================================
        # Construct heterogeneous graph
        # Nodes include:
        # - time nodes
        # - user nodes (time-specific)
        # - item nodes (time-specific)
        # =====================================================
        num_nodes = self.num_times + (self.num_times + 1) * (self.num_users + self.num_items)
        adj_mat = dok_matrix((num_nodes, num_nodes), dtype=np.float32)

        indices = train_data[:][0]

        for n in tqdm(range(len(indices))):
            t_id = indices[n][0]
            u_id = indices[n][1]
            i_id = indices[n][2]

            # Construct time-aware user/item node indices
            u_t_id = (t_id + 1) * (self.num_users + self.num_items) + u_id
            i_t_id = (t_id + 1) * (self.num_users + self.num_items) + self.num_users + i_id

            # Graph edges:
            # user_t ↔ item_t (interaction)
            adj_mat[u_t_id, i_t_id] = 1.0

            # user_t ↔ time
            adj_mat[u_t_id, t_id] = 1.0

            # item_t ↔ time
            adj_mat[i_t_id, t_id] = 1.0

            # global user ↔ time-specific user
            adj_mat[u_id, u_t_id] = 1.0

            # global item ↔ time-specific item
            adj_mat[i_id, i_t_id] = 1.0

        # Enable temporal transitions between adjacent time slices
        for i in range(1, self.num_times):
            adj_mat[i - 1, i] = 1.0

        # Symmetrize adjacency matrix
        adj_mat = adj_mat + adj_mat.T
        adj_mat = adj_mat.tocoo()

        print("--Normalize adjacent matrix...")

        # =====================================================
        # Normalize adjacency matrix (symmetric normalization)
        # A_norm = D^{-1/2} A D^{-1/2}
        # =====================================================
        rowD = np.array(adj_mat.sum(1)).squeeze()
        colD = np.array(adj_mat.sum(0)).squeeze()

        for i in tqdm(range(len(adj_mat.data))):
            adj_mat.data[i] = adj_mat.data[i] / pow(
                rowD[adj_mat.row[i]] * colD[adj_mat.col[i]],
                0.5
            )

        # Convert to PyTorch sparse tensor
        self.adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(adj_mat).coalesce().to(self.device)

        # =====================================================
        # Initialize node embeddings
        # =====================================================
        self.embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(num_nodes, self.embed_dim))
        )

        # MLP for nonlinear interaction modeling
        self.mlp = MultiLayerPerceptron(
            3 * self.embed_dim,
            self.mlp_dims,
            0.1,
            output_layer=False
        )

        # Final prediction layer
        self.fc = nn.Linear(self.mlp_dims[-1] + self.embed_dim, 1)

        # Store embeddings at each propagation layer
        self.E_list = [None] * (self.k + 1)
        self.E_list[0] = self.embedding

        self.E = None

    def forward(self, x, phase='train'):
        """
        Forward pass of GraphModeling.

        Args:
            x: Tensor of shape (batch_size, 3)
               [time_id, user_id, item_id]

        Returns:
            Predicted QoS values
        """

        t_id, u_id, i_id = x[:, 0], x[:, 1], x[:, 2]

        # Map to global graph node indices
        _u_id = u_id + self.num_times
        _i_id = i_id + self.num_times + self.num_users

        # =====================================================
        # GNN propagation (message passing)
        # =====================================================
        for layer in range(1, self.k + 1):
            self.E_list[layer] = torch.spmm(
                sparse_dropout(self.adj_norm, self.dropout),
                self.E_list[layer - 1]
            )

        # Aggregate embeddings from all layers (similar to LightGCN)
        self.E = sum(self.E_list)

        # =====================================================
        # Interaction modeling
        # =====================================================

        # (1) MLP-based nonlinear interaction
        mlp = self.mlp(
            torch.cat([
                self.E[t_id],     # time embedding
                self.E[_u_id],    # user embedding
                self.E[_i_id]     # item embedding
            ], dim=1)
        )

        # (2) Tensor factorization (element-wise product)
        tf = self.E[t_id] * self.E[_u_id] * self.E[_i_id]

        # (3) Combine both interaction signals
        y = self.fc(
            torch.cat([tf, mlp], dim=1)
        ).squeeze(1)

        return y