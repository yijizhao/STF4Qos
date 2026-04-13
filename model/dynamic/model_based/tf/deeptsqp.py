import numpy as np
import torch
from torch import nn, sparse_coo_tensor

from util.similarity import PearsonCorrleationCoefficent


class DeepTemporalAwareServiceQoSPrediction(nn.Module):
    """
    DeepTSQP Model: Temporal-aware Service QoS Prediction

    Reference:
    Zou, Guobing, et al.
    "DeepTSQP: Temporal-aware service QoS prediction via deep neural network and feature integration."
    Knowledge-Based Systems, 2022.

    Core idea:
    - Construct temporal user/item interaction features
    - Incorporate similarity-aware feature enhancement
    - Model temporal dynamics using GRU
    """

    def __init__(self, train_data, field_dims, field_order, config):
        super().__init__()

        # Basic configuration
        self.field_order = field_order
        self.num_times = field_dims[0]
        self.num_users = field_dims[1]
        self.num_items = field_dims[2]

        self.embed_dim = config['embed_dim']
        dropout = config['DeepTSQP']['dropout']
        self.gru_layers = config['DeepTSQP']['gru_layers']
        self.time_window = config['DeepTSQP']['time_window']
        self.device = config['device']

        # =====================================================
        # Build item-item similarity (for user-side feature enhancement)
        # =====================================================
        index = train_data[:][0]
        value = np.array(train_data[:][1])

        tmp = list()
        for t in range(self.num_times):
            # Extract interactions at time t
            ind = np.squeeze(np.argwhere(index[:, 0] == t))
            sub_index = index[ind]
            sub_index = sub_index[:, [2, 1]]  # (item, user)
            sub_value = value[ind]

            # Construct dense matrix: item × user
            tmp.append(
                sparse_coo_tensor(
                    sub_index.T,
                    sub_value.T,
                    [self.num_items, self.num_users]
                ).to_dense()
            )

        # Compute item-item similarity via PCC
        self.sim_ii, _ = PearsonCorrleationCoefficent(
            torch.cat(tmp, dim=1).to(self.device)
        )

        # =====================================================
        # Construct temporal user interaction features
        # =====================================================
        self.user_feature = [None] * (self.num_times + self.time_window)

        y = torch.ones(self.num_users, self.num_items)

        for t in range(len(self.user_feature)):
            if t < self.time_window:
                # Padding for initial time steps
                self.user_feature[t] = torch.zeros(
                    self.num_users, self.num_items
                ).to(self.device)
            else:
                ind = np.squeeze(np.argwhere(index[:, 0] == t))
                sub_index = index[ind]
                sub_index = sub_index[:, [1, 2]]  # (user, item)
                sub_value = value[ind]

                sparse_r = sparse_coo_tensor(
                    sub_index.T,
                    sub_value.T,
                    [self.num_users, self.num_items]
                )

                x = sparse_r.to_dense()

                # Binarize interaction (presence/absence)
                self.user_feature[t] = torch.where(x > 0, y, x).to(self.device)

        # =====================================================
        # Build user-user similarity (for item-side feature enhancement)
        # =====================================================
        tmp = list()
        for t in range(self.num_times):
            ind = np.squeeze(np.argwhere(index[:, 0] == t))
            sub_index = index[ind]
            sub_index = sub_index[:, [1, 2]]  # (user, item)
            sub_value = value[ind]

            tmp.append(
                sparse_coo_tensor(
                    sub_index.T,
                    sub_value.T,
                    [self.num_users, self.num_items]
                ).to_dense()
            )

        # Compute user-user similarity via PCC
        self.sim_uu, _ = PearsonCorrleationCoefficent(
            torch.cat(tmp, dim=1).to(self.device)
        )

        # =====================================================
        # Construct temporal item interaction features
        # =====================================================
        self.item_feature = [None] * (self.num_times + self.time_window)

        y = torch.ones(self.num_items, self.num_users)

        for t in range(len(self.item_feature)):
            if t < self.time_window:
                # Padding
                self.item_feature[t] = torch.zeros(
                    self.num_items, self.num_users
                ).to(self.device)
            else:
                ind = np.squeeze(np.argwhere(index[:, 0] == t))
                sub_index = index[ind]
                sub_index = sub_index[:, [2, 1]]  # (item, user)
                sub_value = value[ind]

                sparse_r = sparse_coo_tensor(
                    sub_index.T,
                    sub_value.T,
                    [self.num_items, self.num_users]
                )

                x = sparse_r.to_dense()

                # Binarization
                self.item_feature[t] = torch.where(x > 0, y, x).to(self.device)

        # =====================================================
        # Temporal modeling with GRU
        # =====================================================
        self.rnn = nn.GRU(
            input_size=self.embed_dim * 2,
            hidden_size=self.embed_dim * 2,
            num_layers=self.gru_layers,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(self.embed_dim * 2, 1)

        # Projection layers for dimensionality reduction
        self.user_projection = nn.Linear(self.num_items, self.embed_dim)
        self.item_projection = nn.Linear(self.num_users, self.embed_dim)

    # =====================================================
    # Temporal prediction for a given sliding window
    # =====================================================
    def predict(self, _user_id, _item_id, slice_id_in_window):
        """
        Perform prediction for a given batch over a temporal window.

        Steps:
        1. Construct user-side features enhanced by item similarity
        2. Construct item-side features enhanced by user similarity
        3. Concatenate features and feed into GRU
        4. Use final hidden state for prediction
        """

        input_seq = list()

        for s in slice_id_in_window:
            # ----- User-side feature -----
            # x_u = user interaction × item similarity
            x_u = torch.index_select(self.user_feature[s], 0, _user_id) * \
                  torch.index_select(self.sim_ii, 0, _item_id)

            # Projection to latent space
            x_u = torch.relu(self.user_projection(x_u))

            # ----- Item-side feature -----
            # x_i = item interaction × user similarity
            x_i = torch.index_select(self.item_feature[s], 0, _item_id) * \
                  torch.index_select(self.sim_uu, 0, _user_id)

            x_i = torch.relu(self.item_projection(x_i))

            # Concatenate user and item features
            input_seq.append(torch.cat([x_u, x_i], dim=-1))

        # Stack into sequence: (batch, time_window+1, embed_dim*2)
        input_seq = torch.stack(input_seq, dim=1)

        # Initialize hidden state
        h0 = torch.zeros(
            self.gru_layers,
            _user_id.shape[0],
            self.embed_dim * 2
        ).to(self.device)

        # GRU forward pass
        _, h = self.rnn(input_seq, h0)

        # Final prediction
        return torch.relu(self.fc(h.squeeze(0))).squeeze(-1)

    # =====================================================
    # Forward pass
    # =====================================================
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 3)
               [time_id, user_id, item_id]

        Returns:
            Predicted QoS values
        """

        time_id, user_id, item_id = x[:, 0], x[:, 1], x[:, 2]

        # Initialize output tensor
        y = torch.zeros(x.shape[0]).to(self.device)

        # Process samples grouped by time (variable-length batching)
        for t in set(time_id.cpu().numpy().tolist()):
            ind = torch.nonzero(torch.eq(time_id, t)).squeeze()

            if ind.ndim == 0:
                ind = ind.unsqueeze(-1)

            # Predict using sliding window [t, t+T]
            sub_y = self.predict(
                user_id[ind],
                item_id[ind],
                list(range(t, t + self.time_window + 1))
            )

            # Assign predictions back
            y.index_put_((ind,), sub_y)

        return y