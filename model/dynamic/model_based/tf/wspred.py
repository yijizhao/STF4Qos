import torch
from torch import nn


class WSPredModel(nn.Module):
    """
    A pytorch implementation of WSPred.
    Reference:
      Yilei Zhang, Zibin Zheng, Michael R. Lyu: WSPred:
      A Time-Aware Personalized QoS Prediction Framework for Web Services. ISSRE 2011: 210-219
    """

    def __init__(self, train_data, field_dims, field_order, config):
        super().__init__()
        embed_dim = config['embed_dim']
        device = config['device']
        self.field_order = field_order
        self.num_times = field_dims[0]
        self.num_users = field_dims[1]
        self.num_items = field_dims[2]
        self.embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(self.num_times + self.num_users + self.num_items, embed_dim)))

    def forward(self, x):
        """
        :param x:  Long tensor of size ``(batch_size, num_fields)``
        """
        # the current time id
        time_id = x[:, 0]
        # the current user id
        user_id = x[:, 1]
        # the current item id
        item_id = x[:, 2]
        time_e = self.embedding[time_id]
        user_e = self.embedding[time_id + self.num_times]
        item_e = self.embedding[item_id + self.num_times + self.num_users]
        y = torch.sum(user_e * item_e * time_e, dim=1)
        return y
