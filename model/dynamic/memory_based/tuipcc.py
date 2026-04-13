import numpy as np
import torch
from torch import nn
from util.similarity import PearsonCorrleationCoefficent


# =========================================================
# Temporal Forecast Module
# =========================================================
class TemporalForecast(nn.Module):
    """
    TemporalForecast captures the historical QoS trend for a given user-item pair.
    Prediction is based on the average of all other observed time slices for the same user-item pair.
    """
    def __init__(self, qos_tensor):
        super().__init__()
        self.qos_tensor = qos_tensor

        # Precompute sum and count along the temporal axis for efficiency
        self.total_sum = qos_tensor.sum(dim=0)
        self.total_cnt = (qos_tensor > 0).sum(dim=0).float()

    def forward(self, user_id, item_id, time_id):
        """
        Forward pass for temporal prediction.

        Args:
            user_id: Tensor of user indices
            item_id: Tensor of item indices
            time_id: Tensor of time indices

        Returns:
            Predicted QoS based on temporal averaging
        """
        curr_val = self.qos_tensor[time_id, user_id, item_id]
        curr_nonzero = (curr_val > 0).float()

        # Exclude the current time slice from aggregation
        sum_others = self.total_sum[user_id, item_id] - curr_val
        cnt_others = self.total_cnt[user_id, item_id] - curr_nonzero

        # Avoid division by zero
        return torch.where(
            cnt_others > 0,
            sum_others / cnt_others,
            torch.zeros_like(sum_others)
        )


# =========================================================
# User-based Collaborative Filtering Module
# =========================================================
class UserCollaborativeFiltering(nn.Module):
    """
    UserCF predicts QoS by aggregating deviations from similar users.
    Uses a time-aggregated Pearson correlation similarity weighted by co-rating statistics.
    """
    def __init__(self, qos_tensor, top_k, device):
        super().__init__()
        self.qos_tensor = qos_tensor
        self.top_k = top_k
        self.device = device

        self.num_users = qos_tensor.size(1)
        self.num_items = qos_tensor.size(2)
        self.num_times = qos_tensor.size(0)

        print("Precomputing user similarities...")

        # Precompute user averages per time slice
        self.user_avg = torch.zeros(self.num_times, self.num_users, device=device)
        for t in range(self.num_times):
            mat = qos_tensor[t]
            cnt = (mat > 0).sum(dim=1).float().clamp(min=1)
            self.user_avg[t] = mat.sum(dim=1) / cnt

        # Precompute time-aggregated user similarities
        self.user_sim_agg = self._compute_agg_similarity()

    # -----------------------------------------------------
    def _compute_agg_similarity(self):
        """
        Compute time-aggregated user similarity matrix.

        Steps:
        1. For each time slice, compute PCC similarity between users
        2. Weight similarity by co-rated items ratio
        3. Aggregate across time slices with log-scaled weighting
        """
        k = self.num_times
        log_k = torch.log(torch.tensor(k + 1, dtype=torch.float, device=self.device))

        weighted_sum = torch.zeros(self.num_users, self.num_users, device=self.device)
        sum_p = torch.zeros_like(weighted_sum)

        for t in range(k):
            mat = self.qos_tensor[t]
            mask = (mat > 0).float()

            # Number of common items rated between users
            common = mask @ mask.T
            nnz_u = mask.sum(dim=1)

            # Weight by co-rated ratio (adjust for small counts)
            w = 2 * common / (nnz_u.unsqueeze(1) + nnz_u.unsqueeze(0) + 1e-8)
            w = torch.nan_to_num(w)

            # Pearson Correlation similarity
            sim_t, _ = PearsonCorrleationCoefficent(mat.detach().cpu())
            sim_t = sim_t.to(self.device)

            weighted_sim_t = w * sim_t

            # Temporal weight
            p_t = common / log_k
            p_t = torch.nan_to_num(p_t)

            weighted_sum += p_t * weighted_sim_t
            sum_p += p_t

        agg = weighted_sum / (sum_p + 1e-8)
        return torch.nan_to_num(agg)

    # -----------------------------------------------------
    def forward(self, user_id, item_id, time_id):
        """
        Forward pass: Predict QoS using top-K similar users.

        Args:
            user_id: Tensor of user indices
            item_id: Tensor of item indices
            time_id: Tensor of time indices

        Returns:
            Predicted QoS tensor
        """
        batch = user_id.size(0)
        R_t = self.qos_tensor[time_id]  # Current time slice

        avg_u = self.user_avg[time_id, user_id]
        sim_u = self.user_sim_agg[user_id]

        # Mask users who have not rated the target item
        item_rated = torch.gather(
            R_t,
            2,
            item_id.view(-1, 1, 1).expand(-1, self.num_users, 1)
        ).squeeze(-1) > 0
        sim_u = torch.where(item_rated, sim_u, torch.zeros_like(sim_u))

        # Top-K neighbors
        top_sim, top_user = sim_u.topk(self.top_k, dim=1)
        top_sim = top_sim / (top_sim.sum(dim=1, keepdim=True) + 1e-8)
        top_sim = torch.nan_to_num(top_sim)

        # Gather neighbors' ratings
        flat_idx = top_user * self.num_items + item_id.unsqueeze(-1)
        r_vs = R_t.view(batch, -1).gather(1, flat_idx)
        avg_v = self.user_avg[time_id.unsqueeze(-1), top_user]

        # Weighted deviation aggregation
        pred = avg_u + (top_sim * (r_vs - avg_v)).sum(dim=1)
        return pred


# =========================================================
# Item-based Collaborative Filtering Module
# =========================================================
class ItemCollaborativeFiltering(nn.Module):
    """
    ItemCF predicts QoS by aggregating deviations from similar items.
    Uses a time-aggregated PCC similarity weighted by co-rated user statistics.
    """
    def __init__(self, qos_tensor, top_k, device):
        super().__init__()
        self.qos_tensor = qos_tensor
        self.top_k = top_k
        self.device = device

        self.num_users = qos_tensor.size(1)
        self.num_items = qos_tensor.size(2)
        self.num_times = qos_tensor.size(0)

        print("Precomputing item similarities...")

        # Precompute item averages per time slice
        self.item_avg = torch.zeros(self.num_times, self.num_items, device=device)
        for t in range(self.num_times):
            mat = qos_tensor[t]
            cnt = (mat > 0).sum(dim=0).float().clamp(min=1)
            self.item_avg[t] = mat.sum(dim=0) / cnt

        # Precompute time-aggregated item similarities
        self.item_sim_agg = self._compute_agg_similarity()

    # -----------------------------------------------------
    def _compute_agg_similarity(self):
        """
        Compute time-aggregated item similarity matrix.

        Steps:
        1. For each time slice, compute PCC similarity between items
        2. Weight similarity by co-rated users ratio
        3. Aggregate across time slices
        """
        k = self.num_times
        log_k = torch.log(torch.tensor(k + 1, dtype=torch.float, device=self.device))

        weighted_sum = torch.zeros(self.num_items, self.num_items, device=self.device)
        sum_p = torch.zeros_like(weighted_sum)

        for t in range(k):
            mat = self.qos_tensor[t]
            mask = (mat > 0).float().T

            common = mask @ mask.T
            nnz_i = mask.sum(dim=1)

            w = 2 * common / (nnz_i.unsqueeze(1) + nnz_i.unsqueeze(0) + 1e-8)
            w = torch.nan_to_num(w)

            sim_t, _ = PearsonCorrleationCoefficent(mat.T.detach().cpu())
            sim_t = sim_t.to(self.device)

            weighted_sim_t = w * sim_t

            p_t = common / log_k
            p_t = torch.nan_to_num(p_t)

            weighted_sum += p_t * weighted_sim_t
            sum_p += p_t

        agg = weighted_sum / (sum_p + 1e-8)
        return torch.nan_to_num(agg)

    # -----------------------------------------------------
    def forward(self, user_id, item_id, time_id):
        """
        Forward pass: Predict QoS using top-K similar items.

        Args:
            user_id: Tensor of user indices
            item_id: Tensor of item indices
            time_id: Tensor of time indices

        Returns:
            Predicted QoS tensor
        """
        batch = user_id.size(0)
        R_t = self.qos_tensor[time_id]

        avg_i = self.item_avg[time_id, item_id]
        sim_i = self.item_sim_agg[item_id]

        # Mask items not rated by the target user
        user_rated = torch.gather(
            R_t,
            1,
            user_id.view(-1, 1, 1).expand(-1, 1, self.num_items)
        ).squeeze(1) > 0
        sim_i = torch.where(user_rated, sim_i, torch.zeros_like(sim_i))

        # Top-K neighbors
        top_sim, top_item = sim_i.topk(self.top_k, dim=1)
        top_sim = top_sim / (top_sim.sum(dim=1, keepdim=True) + 1e-8)
        top_sim = torch.nan_to_num(top_sim)

        # Gather neighbors' ratings
        flat_idx = top_item * self.num_users + user_id.unsqueeze(-1)
        r_sf = R_t.permute(0, 2, 1).reshape(batch, -1).gather(1, flat_idx)
        avg_f = self.item_avg[time_id.unsqueeze(-1), top_item]

        # Weighted deviation aggregation
        pred = avg_i + (top_sim * (r_sf - avg_f)).sum(dim=1)
        return pred


# =========================================================
# Hybrid TUIPCC Forecast Model
# =========================================================
class HybridForecast(nn.Module):
    """
    HybridForecast combines Temporal, UserCF, and ItemCF predictions using
    weighted linear combination.
    """
    def __init__(self, train_data, field_dims, config):
        super().__init__()

        self.num_times = field_dims[0]
        self.num_users = field_dims[1]
        self.num_items = field_dims[2]

        # Hyperparameters
        self.lamda = config['TUIPCC']['lambda']  # weight for user vs. item CF
        self.mu = config['TUIPCC']['mu']         # weight for temporal vs. CF
        self.T = config['TUIPCC']['T']
        self.k = config['TUIPCC']['k']           # number of time slices
        self.top_k = config['TUIPCC']['top_k']   # number of neighbors
        self.device = config['device']

        # Build time-sliced QoS tensor
        qos_tensor, slice_map = self._build_qos_tensor(train_data)

        # Register as buffer for GPU access
        self.register_buffer("qos_tensor", qos_tensor)
        self.register_buffer("slice_map", slice_map)

        # Initialize sub-models
        self.tf = TemporalForecast(qos_tensor)
        self.ucf = UserCollaborativeFiltering(qos_tensor, self.top_k, self.device)
        self.icf = ItemCollaborativeFiltering(qos_tensor, self.top_k, self.device)

    # -----------------------------------------------------
    def _build_qos_tensor(self, train_data):
        """
        Construct a time-sliced QoS tensor from raw training data.

        Steps:
        1. Map original time indices to discrete slices
        2. Aggregate multiple observations per slice by averaging
        """
        indices_list = []
        values_list = []

        for i in range(len(train_data)):
            rec, val = train_data[i]
            indices_list.append(rec)
            values_list.append(val)

        indices = np.array(indices_list)
        values = np.array(values_list)

        # Map original time indices to k slices
        if self.num_times == self.k:
            slice_map = torch.arange(self.k, device=self.device)
        else:
            bin_size = self.num_times / self.k
            slice_map = torch.floor(
                torch.arange(self.num_times, device=self.device) / bin_size
            ).long().clamp(max=self.k - 1)

        # Initialize tensor
        qos = torch.zeros(self.k, self.num_users, self.num_items, device=self.device)
        count = torch.zeros_like(qos)

        for i in range(len(indices)):
            t_orig, u, s = indices[i]
            t_new = slice_map[t_orig]
            qos[t_new, u, s] += values[i]
            count[t_new, u, s] += 1

        # Average repeated entries
        qos = torch.where(count > 0, qos / count, torch.zeros_like(qos))
        return qos, slice_map

    # -----------------------------------------------------
    def forward(self, x):
        """
        Forward pass: combine temporal, user CF, and item CF predictions.

        Args:
            x: Tensor of shape [batch_size, 3] -> [time, user, item]

        Returns:
            Weighted hybrid QoS prediction
        """
        time_id_orig = x[:, 0].long()
        time_id = self.slice_map[time_id_orig]

        user_id = x[:, 1].long()
        item_id = x[:, 2].long()

        # Sub-model predictions
        tf_pred = self.tf(user_id, item_id, time_id)
        ucf_pred = self.ucf(user_id, item_id, time_id)
        icf_pred = self.icf(user_id, item_id, time_id)

        # Combine user CF and item CF
        cf_pred = self.lamda * ucf_pred + (1 - self.lamda) * icf_pred

        # Combine temporal and CF predictions
        final_pred = self.mu * tf_pred + (1 - self.mu) * cf_pred

        return final_pred