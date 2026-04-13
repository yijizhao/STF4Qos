import math
import numpy as np
import torch
import torch.nn as nn
from torch import sparse_coo_tensor
from tqdm import tqdm


def euclidean_dist(x, y):
    return torch.cdist(x, y)


class SCAGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SCAGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_zh, self.W_zx, self.b_z = self.get_three_parameters()
        self.W_rh, self.W_rx, self.b_r = self.get_three_parameters()
        self.W_hh, self.W_hx, self.b_h = self.get_three_parameters()
        self.W_ch, _, self.b_c = self.get_three_parameters()

        self.reset()

    def forward(self, input, state):
        h = state
        c = torch.relu(state @ self.W_ch + self.b_c)

        for x in input:
            x = x.unsqueeze(-1)
            z = torch.sigmoid(h @ self.W_zh + x @ self.W_zx + self.b_z + c)
            r = torch.sigmoid(h @ self.W_rh + x @ self.W_rx + self.b_r + c)
            ht = torch.tanh((h * r) @ self.W_hh + x @ self.W_hx + self.b_h + c)
            h = (1 - z) * h + z * ht

        return h

    def get_three_parameters(self):
        return nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim)), \
               nn.Parameter(torch.FloatTensor(self.input_dim, self.hidden_dim)), \
               nn.Parameter(torch.FloatTensor(self.hidden_dim))

    def reset(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)


class Interaction(nn.Module):
    def __init__(self, input_size):
        super(Interaction, self).__init__()
        self.interaction = nn.Linear(input_size, input_size)
        self.forget_gate = nn.Linear(input_size, input_size)

    def forward(self, a, b):
        x = a + b
        forget_rate = torch.sigmoid(self.forget_gate(x))
        temp = self.interaction(x)
        return forget_rate * temp + (1 - forget_rate) * x


class SpatialContextAwareTimeSeriesForecast(nn.Module):

    def __init__(self, train_data, test_data, field_dims, field_order, config):
        super().__init__()

        self.field_order = field_order
        self.num_times, self.num_users, self.num_items = field_dims

        self.embed_dim = config['embed_dim']
        self.top_k = config['SCATSF']['top_k']
        self.time_window = config['SCATSF']['time_window']
        self.inner_batch = config['SCATSF']['inner_batch']
        self.device = config['device']

        # ===== QoS =====
        self.qos = self.build_qos(train_data)
        self.qos_norm, self.max_qos, self.min_qos = self.normalize_qos(self.qos)
        self.uu_sim, self.ii_sim = self.calculate_similarity(self.qos_norm)

        self._qos = self.complete_qos(
            test_data, self.uu_sim, self.qos_norm,
            self.max_qos, self.min_qos
        )

        # ===== Embedding =====
        self.user_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(self.num_users, self.embed_dim))
        )
        self.item_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(self.num_items, self.embed_dim))
        )

      
        self.user_cnn = nn.Conv2d(1, 1, kernel_size=(self.top_k, 1))
        self.item_cnn = nn.Conv2d(1, 1, kernel_size=(self.top_k, 1))

        self.interaction = Interaction(self.embed_dim)
        self.gru = SCAGRU(input_dim=1, hidden_dim=self.embed_dim)

        self.fc = nn.Linear(4 * self.embed_dim, self.embed_dim)
        self.pred = nn.Linear(self.embed_dim, 1)

    def build_qos(self, train_data):
        index = train_data[:][0]
        value = np.array(train_data[:][1])

        qos = [None] * (self.num_times + self.time_window)

        for t in range(len(qos)):
            if t < self.time_window:
                qos[t] = torch.zeros(self.num_users, self.num_items, device=self.device)
            else:
                ind = np.squeeze(np.argwhere(index[:, 0] == t))
                if ind.size == 0:
                    qos[t] = torch.zeros(self.num_users, self.num_items, device=self.device)
                    continue

                sub_index = index[ind][:, [1, 2]]
                sub_value = value[ind]

                qos[t] = sparse_coo_tensor(
                    sub_index.T,
                    sub_value.T,
                    [self.num_users, self.num_items]
                ).to_dense().to(self.device)

        return qos

    def normalize_qos(self, qos):
        tmp = torch.stack(qos, dim=0).permute(1, 2, 0)

        max_qos = torch.max(tmp, dim=2).values
        tmp2 = torch.where(tmp > 0, tmp, torch.inf)
        min_qos = torch.min(tmp2, dim=2).values
        min_qos = torch.where(torch.isinf(min_qos), 0, min_qos)

        dev_qos = max_qos - min_qos
        dev_qos = torch.where(dev_qos == 0, torch.ones_like(dev_qos), dev_qos)

        norm_qos = []
        for t in range(len(qos)):
            x = (qos[t] - min_qos) / dev_qos
            x = torch.where(torch.isnan(x), 0, x)
            x = torch.clamp(x, min=0.0)
            norm_qos.append(x)

        return norm_qos, max_qos, min_qos

    def calculate_similarity(self, norm_qos):
        X_ut, Y_st = [], []

        for t in range(self.time_window, len(norm_qos)):
            cnt_u = torch.count_nonzero(norm_qos[t], dim=1)
            cnt_u = torch.where(cnt_u == 0, torch.ones_like(cnt_u), cnt_u)
            X_ut.append(torch.sum(norm_qos[t], dim=1) / cnt_u)

            cnt_i = torch.count_nonzero(norm_qos[t], dim=0)
            cnt_i = torch.where(cnt_i == 0, torch.ones_like(cnt_i), cnt_i)
            Y_st.append(torch.sum(norm_qos[t], dim=0) / cnt_i)

        X_ut = torch.stack(X_ut, dim=1)
        Y_st = torch.stack(Y_st, dim=1)

        uu_sim = 1.0 / (1.0 + euclidean_dist(X_ut, X_ut))
        ii_sim = 1.0 / (1.0 + euclidean_dist(Y_st, Y_st))

        return uu_sim.to(self.device), ii_sim.to(self.device)

    def predict(self, user_id, item_id, sim, qos_norm, dev_qos, qos_min):
        uu_sim = sim[user_id, :]

        rates = qos_norm[:, item_id].T
        uu_sim = uu_sim.index_put(
            torch.nonzero(rates == 0, as_tuple=True),
            torch.tensor(0., device=uu_sim.device)
        )

        top_sim, top_user = uu_sim.topk(self.top_k)
        top_sim = top_sim / (top_sim.sum(dim=1, keepdim=True) + 1e-8)

        rate = qos_norm[top_user, item_id.unsqueeze(-1)]
        rate = torch.sum(top_sim * rate, dim=1)

        idx = user_id * self.num_items + item_id
        return qos_min.view(-1)[idx] + dev_qos.view(-1)[idx] * rate

    def complete_qos(self, test_data, uu_sim, qos_norm, qos_max, qos_min):
        index = test_data[:][0]
        dev_qos = qos_max - qos_min

        _qos = []

        for t in tqdm(range(len(qos_norm))):
            if t < self.time_window:
                _qos.append(torch.zeros(self.num_users, self.num_items, device=self.device))
            else:
                ind = np.squeeze(np.argwhere(index[:, 0] == t))
                if ind.size == 0:
                    _qos.append(self.qos[t])
                    continue

                index_t = torch.from_numpy(index[ind]).to(self.device)

                value = self.predict(
                    index_t[:, 1],
                    index_t[:, 2],
                    uu_sim,
                    qos_norm[t],
                    dev_qos,
                    qos_min
                )

                tmp = sparse_coo_tensor(
                    index_t[:, 1:3].t(),
                    value,
                    [self.num_users, self.num_items]
                ).to_dense().to(self.device)

                _qos.append(self.qos[t] + tmp)

        return _qos

    def forward(self, x):
        time_id, user_id, item_id = x[:, 0], x[:, 1], x[:, 2]

        y = torch.zeros(x.shape[0], device=self.device)

        for t in set(time_id.cpu().numpy().tolist()):
            ind = torch.nonzero(time_id == t).squeeze()
            if ind.ndim == 0:
                ind = ind.unsqueeze(-1)

            u = user_id[ind]
            i = item_id[ind]

            _, top_u = self.uu_sim[u].topk(self.top_k)
            _, top_i = self.ii_sim[i].topk(self.top_k)

            u_nei = self.user_embedding[top_u].unsqueeze(1)
            u_nei = self.user_cnn(u_nei).squeeze(2).squeeze(1)

            i_nei = self.item_embedding[top_i].unsqueeze(1)
            i_nei = self.item_cnn(i_nei).squeeze(2).squeeze(1)

            u_emb = self.user_embedding[u]
            i_emb = self.item_embedding[i]

            cross = torch.cat([
                self.interaction(u_emb, i_emb),
                self.interaction(u_emb, u_nei),
                self.interaction(u_nei, i_emb),
                self.interaction(u_nei, i_nei)
            ], dim=1)

            hidden = self.fc(cross)

            seq = []
            for s in range(t, t + self.time_window + 1):
                seq.append(self._qos[s][u, i])

            seq = torch.stack(seq, dim=0)

            out = torch.relu(self.gru(seq, hidden))
            out = self.pred(out)

            y[ind] = out.squeeze(-1)

        return y