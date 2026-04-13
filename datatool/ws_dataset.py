"""
@author Hao Wu
@email haowu@ynu.edu.cn
@date  July 17, 2023
@version 0.1

"""
import numpy as np
import pandas as pd
import torch
from scipy.sparse import dok_matrix
from torch.nn.utils.rnn import pad_sequence

from datatool.qos_dataset import QoSDataset


class WSDatasetWithoutContext(QoSDataset):
    """
      The dataset only considers the fields of response-time, throughput, user ID, service ID
    """

    def __init__(self, dataset_path, data_task='BOTH'):
        """ read and process the dataset with pandas"""
        print("WSDatasetWithoutContext...")
        self.df = pd.DataFrame(pd.read_csv(dataset_path))
        self.data_task = data_task.lower()
        self.rt = self.df["RT"].astype(np.float32)
        self.tp = self.df["TP"].astype(np.float32)
        self.items = self.df["Service_ID"].astype(np.int_)
        self.users = self.df["User_ID"].astype(np.int_)
        """stack the arrays to form the full one """
        self.records = np.column_stack((self.users, self.items))

    def field_dims(self):
        """
        calculate the field dimensions, a.k.a, the total number of field objects
        """
        return [np.max(self.users) + 1, np.max(self.items) + 1]

    def field_order(self):
        """
        in default case, list[0][...] is for users and list[1][...] is for items
        """
        return [[0], [1]]

    def dataset_fields(self):
        """
        return readable field names
        """
        real_order = []
        for i in self.field_order():
            real_order.append([j + 2 for j in i])
        return [list(self.df.columns[i]) for i in real_order]

    def split_field(self):
        """
        return the fields to enable field-aware random split
        """
        return self.users.tolist()

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, index):
        match self.data_task:
            case self.TASK_RT:
                return self.records[index], self.rt[index]
            case self.TASK_TP:
                return self.records[index], self.tp[index]
            case self.TASK_BOTH:
                return self.records[index], self.rt[index], self.tp[index]
            case '_':
                return self.records[index], self.rt[index], self.tp[index]


class WSDatasetWithContext(QoSDataset):
    """
          The dataset also considers other fields besides response-time, throughput, user ID, service ID
    """

    def __init__(self, dataset_path, data_task='BOTH'):
        """ read and process the dataset with pandas"""
        print("WSDatasetWithContext...")
        self.df = pd.DataFrame(pd.read_csv(dataset_path))
        self.data_task = data_task.lower()
        self.rt = self.df["RT"].astype(np.float32)
        self.tp = self.df["TP"].astype(np.float32)
        self.items = self.df["Service_ID"].astype(np.int_)
        self.users = self.df["User_ID"].astype(np.int_)
        """read the remaining columns as numpy array"""
        self.remains = self.df[self.df.columns[4:len(self.df.columns)]]
        """stack the arrays to form the full one """
        self.records = np.column_stack((self.users, self.items, self.remains.astype(np.int_)))
        self.user_context = self.df[['User_ID', 'User_IP', 'User_Country', 'User_AS', 'User_GP']].astype(np.int_)
        self.user_context.drop_duplicates(inplace=True, ignore_index=True)

        self.item_context = self.df[
            ['Service_ID', 'Service_Provider', 'Service_IP', 'Service_Country', 'Service_AS', 'Service_GP']].astype(
            np.int_)
        self.item_context.drop_duplicates(inplace=True, ignore_index=True)

    def field_dims(self):
        """
        calculate the field dimensions, a.k.a, the total number of field objects
        """
        return [np.max(self.records[:, i]) + 1 for i in range(self.records.shape[1])]

    def field_order(self):
        """
        in default case, list[0][...] is for users and list[1][...] is for items
        """
        return [[0, 2, 3, 4, 5], [1, 6, 7, 8, 9, 10]]

    def dataset_fields(self):
        """
        return readable field names
        """
        real_order = []
        for i in self.field_order():
            real_order.append([j + 2 for j in i])
        return [list(self.df.columns[i]) for i in real_order]

    def split_field(self):
        """
        return the fields to enable field-aware random split
        """
        return self.users.tolist()

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, index):
        match self.data_task:
            case self.TASK_RT:
                return self.records[index], self.rt[index]
            case self.TASK_TP:
                return self.records[index], self.tp[index]
            case self.TASK_BOTH:
                return self.records[index], self.rt[index], self.tp[index]
            case '_':
                return self.records[index], self.rt[index], self.tp[index]


class WSDatasetWithTopology(QoSDataset):
    """
      The dataset also considers topology of as besides response-time, throughput, user ID, service ID
    """

    def __init__(self, dataset_path, data_task='BOTH'):
        """ read and process the dataset with pandas"""
        print("WSDatasetWithTopology...")
        self.df = pd.DataFrame(pd.read_csv(dataset_path))
        self.data_task = data_task.lower()
        self.rt = self.df["RT"].astype(np.float32)
        self.tp = self.df["TP"].astype(np.float32)
        self.users = self.df["User_ID"].astype(np.int_)
        self.items = self.df["Service_ID"].astype(np.int_)
        self.num_users = np.max(self.users) + 1
        self.num_items = np.max(self.items) + 1

        print("--Re-assign identifier for users, services, path nodes...")
        self.users = self.users + 1  # id 0 is remained for padding usage
        self.items = self.items + 1 + self.num_users
        self.paths = self.df["ASPath"].apply(lambda x: x.split('-'))
        self.new_paths = []
        self.path_lens = []
        self.as_set = set()
        offset = self.num_users + self.num_items
        for i in range(len(self.paths)):
            self.as_set.update(self.paths[i])
            path_i = [int(as_id) + offset for as_id in self.paths[i]]
            path_i.insert(0, self.users[i])
            path_i.append(self.items[i])
            self.new_paths.append(path_i)
            self.path_lens.append(len(path_i))
        # print(self.new_paths)
        print("--Build adjacent matrix from paths...")
        num_nodes = len(self.as_set) + offset + 1
        self.adj_mat = dok_matrix((num_nodes, num_nodes), dtype=np.float32)
        # enable symmetric graph
        for path in self.new_paths:
            for i in range(len(path) - 1):
                if (path[i], path[i + 1]) in self.adj_mat:
                    continue
                else:
                    self.adj_mat[path[i], path[i + 1]] = 1.0
                    self.adj_mat[path[i + 1], path[i]] = 1.0
        for i in range(self.adj_mat.shape[0]):
            self.adj_mat[i, i] = 1.0  # enable self-loop
        print("--Pad node sequences of paths...")
        self.paths = list(map(lambda x: torch.tensor(x), self.new_paths))
        # print(self.paths[0:5])
        self.paths = pad_sequence(self.paths, batch_first=True)
        # print(self.paths[:,0:5])
        # print(self.paths)

        """stack the arrays to form the full one """
        self.records = torch.column_stack((torch.tensor(self.users),
                                           torch.tensor(self.items),
                                           self.paths,
                                           torch.tensor(self.path_lens)))

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, index):
        match self.data_task:
            case self.TASK_RT:
                return self.records[index], self.rt[index]
            case self.TASK_TP:
                return self.records[index], self.tp[index]
            case self.TASK_BOTH:
                return self.records[index], self.rt[index], self.tp[index]
            case '_':
                return self.records[index], self.rt[index], self.tp[index]

    def field_dims(self):
        """
        calculate the field dimensions, a.k.a, the total number of field objects
        """
        return [self.num_users, self.num_items, len(self.as_set)]

    def field_order(self):
        """
        in default case, list[0][...] is for users and list[1][...] is for items, list[2][...] is for paths
        """
        return [[0], [1], [2]]

    def dataset_fields(self):
        """
        return readable field names
        """
        real_order = []
        for i in self.field_order():
            real_order.append([j + 2 for j in i])
        return [list(self.df.columns[i]) for i in real_order]

    def split_field(self):
        """
        return the fields to enable field-aware random split
        """
        return self.users.tolist()

    def adjacent_matrix(self):
        return self.adj_mat.tocoo(True)


class WSDatasetWithContextTopology(QoSDataset):
    """
      The dataset also considers topology of as besides response-time, throughput, user ID, service ID
    """

    def __init__(self, dataset_path, data_task='BOTH'):
        """ read and process the dataset with pandas"""
        print("WSDatasetWithContextTopology...")
        self.df = pd.DataFrame(pd.read_csv(dataset_path))
        self.data_task = data_task.lower()
        self.rt = self.df["RT"].astype(np.float32)
        self.tp = self.df["TP"].astype(np.float32)
        self.users = self.df["User_ID"].astype(np.int_)
        self.items = self.df["Service_ID"].astype(np.int_)
        self.num_users = np.max(self.users) + 1
        self.num_items = np.max(self.items) + 1

        print("--Re-assign identifier for users, services, path nodes...")
        self.users = self.users + 1  # id 0 is remained for padding usage
        self.items = self.items + 1 + self.num_users
        self.paths = self.df["ASPath"].apply(lambda x: x.split('-'))
        self.new_paths = []
        self.path_lens = []
        self.as_set = set()
        offset = self.num_users + self.num_items
        for i in range(len(self.paths)):
            self.as_set.update(self.paths[i])
            path_i = [int(as_id) + offset for as_id in self.paths[i]]
            path_i.insert(0, self.users[i])
            path_i.append(self.items[i])
            self.new_paths.append(path_i)
            self.path_lens.append(len(path_i))
        # print(self.new_paths)
        print("--Build adjacent matrix from paths...")
        num_nodes = len(self.as_set) + offset + 1
        self.adj_mat = dok_matrix((num_nodes, num_nodes), dtype=np.float32)
        # enable symmetric graph
        for path in self.new_paths:
            for i in range(len(path) - 1):
                if (path[i], path[i + 1]) in self.adj_mat:
                    continue
                else:
                    self.adj_mat[path[i], path[i + 1]] = 1.0
                    self.adj_mat[path[i + 1], path[i]] = 1.0
        for i in range(self.adj_mat.shape[0]):
            self.adj_mat[i, i] = 1.0  # enable self-loop
        print("--Pad node sequences of paths...")
        self.paths = list(map(lambda x: torch.tensor(x), self.new_paths))
        # print(self.paths[0:5])
        self.paths = pad_sequence(self.paths, batch_first=True)
        # print(self.paths[:,0:5])
        # print(self.paths)
        """stack the arrays to form the full one """
        self.records = torch.column_stack((torch.tensor(self.users),
                                           torch.tensor(self.items),
                                           self.paths,
                                           torch.tensor(self.path_lens)))

        """read the remaining columns as numpy array"""
        self.user_context = self.df[['User_ID', 'User_IP', 'User_Country', 'User_AS', 'User_GP']].astype(np.int_)
        self.user_context.drop_duplicates(inplace=True, ignore_index=True)

        self.item_context = self.df[
            ['Service_ID', 'Service_Provider', 'Service_IP', 'Service_Country', 'Service_AS', 'Service_GP']].astype(
            np.int_)
        self.item_context.drop_duplicates(inplace=True, ignore_index=True)

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, index):
        match self.data_task:
            case self.TASK_RT:
                return self.records[index], self.rt[index]
            case self.TASK_TP:
                return self.records[index], self.tp[index]
            case self.TASK_BOTH:
                return self.records[index], self.rt[index], self.tp[index]
            case '_':
                return self.records[index], self.rt[index], self.tp[index]

    def field_dims(self):
        """
        calculate the field dimensions, a.k.a, the total number of field objects
        """
        return [self.num_users, self.num_items, len(self.as_set)]

    def field_order(self):
        """
        in default case, list[0][...] is for users and list[1][...] is for items, list[2][...] is for paths
        """
        return [[0], [1], [2]]

    def dataset_fields(self):
        """
        return readable field names
        """
        real_order = []
        for i in self.field_order():
            real_order.append([j + 2 for j in i])
        return [list(self.df.columns[i]) for i in real_order]

    def split_field(self):
        """
        return the fields to enable field-aware random split
        """
        return self.users.tolist()

    def adjacent_matrix(self):
        return self.adj_mat.tocoo(True)


if __name__ == '__main__':
    """
    dataset = WSDatasetWithContext("../dataset/ws-topo/rt_tp_ctx.csv", 'RT')
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    # print(dataset.split_field())
    print(dataset.user_context)
    print(dataset.item_context)
    """
    """
    dataset = WSDatasetWithoutContext("../dataset/ws-topo/rt_tp_ctx.csv", 'RT')
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    # print(dataset.split_field())
   

    dataset = WSDatasetWithTopology("../dataset/ws-topo/rt_tp_topo.csv", 'RT')
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    print(dataset.adjacent_matrix())
    # print(dataset.records[:, 2:])
    # print(sum(dataset.field_dims()))
    # print(dataset.split_field())
    """
    dataset = WSDatasetWithContextTopology("../dataset/ws-topo/rt_tp_ctx_topo.csv", 'RT')
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    print(dataset.adjacent_matrix())
    # print(dataset.records[:, 2:])
    # print(sum(dataset.field_dims()))
    # print(dataset.split_field())
