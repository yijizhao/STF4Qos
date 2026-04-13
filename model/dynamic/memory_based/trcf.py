import torch
import torch.nn as nn
import numpy as np
import logging


class TRCFModel(nn.Module):
    """
    TRCF (Time-aware Regularized Collaborative Filtering) Model
    ------------------------------------------------------------
    A fully GPU-accelerated implementation designed for efficient QoS prediction.

    Core idea:
    - Construct a 3D QoS tensor (user × service × time)
    - Compute user-user similarity (PCC-based)
    - Compute service-service similarity (min-based co-occurrence similarity)
    - Predict missing QoS via neighborhood-based deviation aggregation
    """

    def __init__(self, train_data, field_dims, field_order, config):
        super().__init__()

        # Basic data and configuration
        self.train_data = train_data
        self.field_dims = field_dims
        self.field_order = field_order
        self.config = config

        # Dimensional settings
        self.num_times = field_dims[0]
        self.num_users = field_dims[1]
        self.num_services = field_dims[2]

        # Hyperparameters
        self.theta_pcc = config.get("theta_pcc", 0.5)        # threshold for user similarity
        self.theta_rbs = config.get("theta_rbs", 0.68)       # threshold for service similarity
        self.alpha = config.get("alpha", 0.5)                # fusion weight (if used)
        self.window_size = config.get("window_size", 64)     # temporal window size
        self.density_threshold = config.get("density_threshold", 0.1)
        self.max_neighbors = config.get("max_neighbors", 50) # number of nearest neighbors

        # Device configuration (GPU preferred)
        self.device = config.get(
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize QoS tensor and mask tensor
        # qos_matrix: stores observed QoS values
        # mask_matrix: indicates whether an entry is observed
        self.qos_matrix = torch.zeros(
            (self.num_users, self.num_services, self.num_times),
            device=self.device
        )
        self.mask_matrix = torch.zeros(
            (self.num_users, self.num_services, self.num_times),
            dtype=torch.bool,
            device=self.device
        )

        # Build tensors and precompute statistics
        self._build_from_training_data()


    def _build_from_training_data(self):
        """
        Construct QoS tensor and auxiliary statistics from training data.

        Outputs:
        - qos_matrix: dense QoS tensor
        - mask_matrix: observation indicator tensor
        - avg_qos: average QoS per (user, service)
        - user_sim: user-user similarity matrix
        - service_sim: service-service similarity matrix
        """

        # Support both DataLoader subset and raw dataset
        if hasattr(self.train_data, "dataset"):
            indices = self.train_data.indices
            dataset = self.train_data.dataset
            records = dataset.records[indices]
            targets = dataset.target[indices]
        else:
            records = self.train_data.records
            targets = self.train_data.target

        # Populate QoS tensor
        for rec, val in zip(records, targets):
            t, u, s = map(int, rec[:3])
            self.qos_matrix[u, s, t] = val
            self.mask_matrix[u, s, t] = True

        # Compute average QoS per (user, service)
        self.avg_qos = torch.zeros(
            (self.num_users, self.num_services),
            device=self.device
        )

        valid_counts = self.mask_matrix.sum(dim=2)  # number of observations
        sum_values = (self.qos_matrix * self.mask_matrix).sum(dim=2)

        # Avoid division by zero
        self.avg_qos[valid_counts > 0] = (
            sum_values[valid_counts > 0] /
            valid_counts[valid_counts > 0]
        )

        # Precompute similarity matrices
        self.user_sim = self._compute_user_similarity()
        self.service_sim = self._compute_service_similarity()

        logging.info(
            f"TRCF-Fast built: {self.num_users} users, {self.num_services} services"
        )


    def _compute_user_similarity(self):
        """
        Compute user-user similarity using a Pearson Correlation-like formulation.

        Steps:
        1. Flatten QoS tensor into (user × features)
        2. Compute mean-centered vectors
        3. Compute cosine similarity on centered data (equivalent to PCC)
        """

        qos_flat = self.qos_matrix.view(self.num_users, -1)
        mask_flat = self.mask_matrix.view(self.num_users, -1)

        # Compute mean QoS for each user
        mean = torch.zeros(self.num_users, device=self.device)
        for u in range(self.num_users):
            valid = mask_flat[u]
            if valid.any():
                mean[u] = qos_flat[u, valid].mean()

        # Mean-centering
        centered = torch.where(
            mask_flat,
            qos_flat - mean.unsqueeze(1),
            torch.zeros_like(qos_flat)
        )

        # Compute covariance matrix
        cov = centered @ centered.T

        # Compute normalization term
        norm = torch.sqrt((centered ** 2).sum(dim=1, keepdim=True))
        denom = norm @ norm.T

        # Final similarity (avoid division by zero)
        sim = torch.where(denom > 0, cov / denom, torch.zeros_like(cov))

        # Self-similarity = 1
        sim.fill_diagonal_(1.0)

        return sim


    def _compute_service_similarity(self):
        """
        Compute service-service similarity based on co-observed QoS values.

        Strategy:
        - For each pair of services, find common observed entries
        - Use the mean of element-wise minimum values as similarity

        Note:
        This is a heuristic similarity (not standard cosine/PCC),
        often used in QoS-aware service computing.
        """

        sim = torch.zeros(
            (self.num_services, self.num_services),
            device=self.device
        )

        for s1 in range(self.num_services):
            vec1 = self.qos_matrix[:, s1, :].reshape(self.num_users, -1)
            mask1 = self.mask_matrix[:, s1, :].reshape(self.num_users, -1)

            for s2 in range(s1, self.num_services):
                vec2 = self.qos_matrix[:, s2, :].reshape(self.num_users, -1)
                mask2 = self.mask_matrix[:, s2, :].reshape(self.num_users, -1)

                # Find common observed entries
                common = mask1 & mask2

                if common.any():
                    # Use mean of element-wise minimum as similarity
                    vmin = torch.min(vec1[common], vec2[common])
                    val = vmin.mean()
                    sim[s1, s2] = sim[s2, s1] = val

        sim.fill_diagonal_(1.0)
        return sim


    def forward(self, x):
        """
        Forward pass: QoS prediction

        Input:
        - x: Tensor of shape [batch_size, 3]
             (time_id, user_id, service_id)

        Output:
        - preds: Predicted QoS values
        """

        time_ids = x[:, 0].long()
        user_ids = x[:, 1].long()
        service_ids = x[:, 2].long()

        batch_size = len(user_ids)

        # Baseline: average QoS of (user, service)
        base_qos = self.avg_qos[user_ids, service_ids]

        # Initialize deviation aggregation
        dev_sum = torch.zeros(batch_size, device=self.device)
        sim_sum = torch.zeros_like(dev_sum)

        # Neighborhood-based collaborative filtering
        for i in range(batch_size):
            u = user_ids[i].item()
            s = service_ids[i].item()
            t = time_ids[i].item()

            # Get top-K similar users
            sims = self.user_sim[u]
            topk_val, topk_idx = torch.topk(sims, self.max_neighbors)

            # Select neighbors who have observed QoS at (s, t)
            valid_mask = self.mask_matrix[topk_idx, s, t]
            valid_users = topk_idx[valid_mask]
            valid_sims = topk_val[valid_mask]

            if len(valid_users) == 0:
                continue

            # Compute deviation from average QoS
            avg_v_s = self.avg_qos[valid_users, s]
            dev_v_s = self.qos_matrix[valid_users, s, t] - avg_v_s

            # Weighted aggregation
            dev_sum[i] = (valid_sims * dev_v_s).sum()
            sim_sum[i] = valid_sims.sum()

        # Normalize deviation
        deviation = torch.zeros_like(base_qos)
        valid = sim_sum > 0
        deviation[valid] = dev_sum[valid] / sim_sum[valid]

        # Final prediction = baseline + deviation
        preds = base_qos + deviation

        # Ensure non-negative QoS
        preds = torch.clamp(preds, min=0)

        return preds