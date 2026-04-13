import numpy as np
import torch
from torch import nn, sparse_coo_tensor
from model.layers.fmlayer import MultiLayerPerceptron

class SteadyandTransientFactorization(nn.Module):
    def __init__(self, train_data, field_dims, field_order, config):
        super().__init__()
        self.field_order = field_order
        self.num_times = field_dims[0]
        self.num_users = field_dims[1]
        self.num_items = field_dims[2]
        self.embed_dim = config['embed_dim']
        self.mlp_dims = config['STF']['mlp_dims']
        self.dropout = config['STF']['dropout']
        self.num_layers = config['STF']['num_layers']
        self.device = config['device']
        index = train_data[:][0]
        value = np.array(train_data[:][1])
        self.dense_r = [None] * self.num_times
        for i in range(self.num_times):
            ind = np.squeeze(np.argwhere(index[:, 0] == i))
            sub_index = index[ind]
            sub_index = sub_index[:, [1, 2]]
            sub_value = value[ind]
            sparse_r = sparse_coo_tensor(sub_index.T, sub_value.T, [self.num_users, self.num_items])
            self.dense_r[i] = sparse_r.to_dense().to(self.device)
        self.all_dense = torch.stack(self.dense_r, dim=0).to(self.device)
        self.conv_u = nn.Conv2d(in_channels=1, out_channels=self.embed_dim, kernel_size=(self.num_times, 1)).to(self.device)
        self.conv_i = nn.Conv2d(in_channels=1, out_channels=self.embed_dim, kernel_size=(self.num_times, 1)).to(self.device)
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten(),nn.Linear(self.embed_dim, self.embed_dim)).to(self.device)
        self.global_embedding_u2=nn.Parameter(torch.zeros(self.num_users, self.embed_dim))
        self.global_embedding_i2=nn.Parameter(torch.zeros(self.num_items, self.embed_dim))
        self.t_embedding = nn.Parameter(torch.zeros(self.num_times, self.embed_dim))
        self.mlp1 = MultiLayerPerceptron(self.embed_dim * 3, self.mlp_dims, self.dropout).to(self.device)
        self.mlp2 = MultiLayerPerceptron(self.embed_dim * 4, self.mlp_dims, self.dropout).to(self.device)
        self.mlp3 = MultiLayerPerceptron(self.embed_dim * 5, self.mlp_dims, self.dropout).to(self.device)
        self.mlp4 = MultiLayerPerceptron(self.embed_dim * 2, self.mlp_dims, self.dropout).to(self.device)
        self.mlp5 = MultiLayerPerceptron(self.embed_dim * 2, self.mlp_dims, self.dropout).to(self.device)
        self.mlp6 = MultiLayerPerceptron(self.embed_dim * 4, self.mlp_dims, self.dropout).to(self.device)
        self.mlp7 = MultiLayerPerceptron(self.embed_dim * 3, self.mlp_dims, self.dropout).to(self.device)
        self.mlp8 = MultiLayerPerceptron(self.embed_dim * 4, self.mlp_dims, self.dropout).to(self.device)
        self.mlp9 = MultiLayerPerceptron(self.embed_dim * 3, self.mlp_dims, self.dropout).to(self.device)
        self.fc_y1 = nn.Linear(self.mlp_dims[-1], 1)
        self.fc2 = nn.Linear(self.mlp_dims[-1], 1)
        self.fc3 = nn.Linear(self.mlp_dims[-1], 1)
        self.fc4 = nn.Linear(self.mlp_dims[-1], 1)
        self.fc5 = nn.Linear(self.mlp_dims[-1], 1)
        self.fc6 = nn.Linear(self.mlp_dims[-1], 1)
        self.fc7 = nn.Linear(self.mlp_dims[-1], 1)
        self.fc8 = nn.Linear(self.mlp_dims[-1], 1)
        self.fc9 = nn.Linear(self.mlp_dims[-1], 1)
        self.fc_y2 = nn.Linear(2,1)
        self.fc_y3 = nn.Linear(3,1)
        self.fc_y4 = nn.Linear(3,1)
        self.gamma1 = nn.Parameter(torch.tensor(0.25))
        self.gamma2 = nn.Parameter(torch.tensor(0.25))
        self.gamma3 = nn.Parameter(torch.tensor(0.25))
        self.gamma4 = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)`` b,t,u,i t=64,u=142,i=4500
        """
        q_u = self.all_dense.permute(1, 0, 2).unsqueeze(1)  # [U, 1, T, I]
        q_i = self.all_dense.permute(2, 0, 1).unsqueeze(1)  # [I, 1, T, U]
        global_embedding_u1 = self.global_pool(self.conv_u(q_u).permute(0, 1, 3, 2))  # [U, D]
        global_embedding_i1 = self.global_pool(self.conv_i(q_i).permute(0, 1, 3, 2))  # [I, D]

        t_id, u_id, i_id = x[:, 0], x[:, 1], x[:, 2]
        u1_g_emb = global_embedding_u1[u_id] #[batch,D]
        u2_g_emb = self.global_embedding_u2[u_id] #[batch,D]
        i1_g_emb = global_embedding_i1[i_id] #[batch,D]
        i2_g_emb = self.global_embedding_i2[i_id] #[batch,D]

        t_emb = self.t_embedding[t_id]

        u_l_emb = u1_g_emb+u2_g_emb+t_emb #[batch,D]
        i_l_emb = i1_g_emb+i2_g_emb+t_emb #[batch,D]

        z1 = self.mlp1(torch.cat([u_l_emb,i_l_emb,t_emb], dim=1))
        y_ust1 = self.fc_y1(z1).squeeze(-1)
        z2 = self.mlp2(torch.cat([u1_g_emb,u2_g_emb,i1_g_emb,i2_g_emb], dim=1))
        y_us = self.fc2(z2)
        z3 = self.mlp3(torch.cat([u1_g_emb,u2_g_emb,i1_g_emb,i2_g_emb,t_emb], dim=1))
        b_us = self.fc3(z3)
        y_ust2= self.fc_y2(torch.cat([y_us,b_us], dim=1)).squeeze(-1)
        z4 = self.mlp4(torch.cat([u1_g_emb,u2_g_emb], dim=1))
        y_u = self.fc4(z4)
        z5 = self.mlp5(torch.cat([i1_g_emb,i2_g_emb], dim=1))
        y_s = self.fc5(z5)
        z6 = self.mlp6(torch.cat([u1_g_emb,u2_g_emb,i1_g_emb,i2_g_emb], dim=1))
        b_u1 = self.fc6(z6)
        z7 = self.mlp7(torch.cat([u1_g_emb,u2_g_emb,t_emb], dim=1))
        b_u2 = self.fc7(z7)
        y_ust3 = self.fc_y3(torch.cat([y_u,b_u1,b_u2], dim=1)).squeeze(-1)
        z8 = self.mlp8(torch.cat([i1_g_emb,i2_g_emb,u1_g_emb,u2_g_emb], dim=1))
        b_s1 = self.fc8(z8)
        z9 = self.mlp9(torch.cat([i1_g_emb,i2_g_emb,t_emb], dim=1))
        b_s2 = self.fc9(z9)
        y_ust4 = self.fc_y4(torch.cat([y_s,b_s1,b_s2], dim=1)).squeeze(-1)
        y_qos = self.gamma1 * y_ust1 + self.gamma2 * y_ust2 + self.gamma3 * y_ust3 + self.gamma4 * y_ust4
        return y_qos