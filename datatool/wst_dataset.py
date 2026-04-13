"""
@author Hao Wu
@email haowu@ynu.edu.cn
@date  July 17, 2023
@version 0.1

"""
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from datatool.qos_dataset import QoSDataset


class WSTDatasetWithoutContext(QoSDataset):
    """
      The dataset  considers only the fields of response-time, slice ID, user ID, service ID
    """

    def __init__(self, dataset_path, data_task='RT'):
        """ read and process the dataset with pandas"""
        print("WSTDatasetWithoutContext...")
        self.df = pd.DataFrame(pd.read_csv(dataset_path))
        self.data_task = data_task.lower()
        self.target = self.df["RT"].astype(np.float32)
        self.times = self.df["Slice_ID"].astype(np.int_)
        self.items = self.df["Service_ID"].astype(np.int_)
        self.users = self.df["User_ID"].astype(np.int_)
        """stack the arrays to form the full one """
        self.records = np.column_stack((self.times, self.users, self.items))

    def field_dims(self):
        """
        calculate the field dimensions, a.k.a, the total number of field objects
        """
        return [np.max(self.times) + 1, np.max(self.users) + 1, np.max(self.items) + 1]

    def field_order(self):
        """
        in default case, list[0][...] is for time_slices, list[1][...] is for users, list[2][...]  is for items
        """
        return [[0], [1], [2]]

    def dataset_fields(self):
        """
        return readable field names
        """
        real_order = []
        for i in self.field_order():
            real_order.append([j + 1 for j in i])
        return [list(self.df.columns[i]) for i in real_order]

    def split_field(self):
        """
        return the fields to enable field-aware random split
        """
        return self.times.astype(np.int_).tolist()

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, index):
        return self.records[index], self.target[index]


class WSTDatasetWithContext(QoSDataset):
    """
          The dataset also considers other fields besides response-time, throughput, user ID, service ID
    """

    def __init__(self, dataset_path, data_task='RT'):
        """ read and process the dataset with pandas"""
        print("WSTDatasetWithContext...")
        self.df = pd.DataFrame(pd.read_csv(dataset_path))
        self.data_task = data_task.lower()
        self.target = self.df["RT"].astype(np.float32)
        self.times = self.df["Slice_ID"].astype(np.int_)
        self.items = self.df["Service_ID"].astype(np.int_)
        self.users = self.df["User_ID"].astype(np.int_)
        """read the remaining columns as numpy array"""
        self.remains = self.df[self.df.columns[4:len(self.df.columns)]]
        """stack the arrays to form the full one """
        self.records = np.column_stack((self.times, self.users, self.items, self.remains.astype(np.int_)))

    def field_dims(self):
        """
        calculate the field dimensions, a.k.a, the total number of field objects
        """
        return [np.max(self.records[:, i]) + 1 for i in range(self.records.shape[1])]

    def field_order(self):
        """
        in default case, list[0][...] is for time_slices, list[1][...] is for users, list[2][...]  is for items
        0:Slice_ID, 1:User_ID, 2:Service_ID, 3:User_IP, 4:User_Country, 5:User_State, 6: User_City, 7:User_GP, 8:User_AS,
        9:Service_IP, 10:Service_Country, 11:Service_State, 12:Service_City, 13:Service_GP, 14:Service_AS
        """
        return [[0], [1, 3, 4, 5, 6, 7, 8], [2, 9, 10, 11, 12, 13, 14]]

    def dataset_fields(self):
        """
        return readable field names
        """
        real_order = []
        for i in self.field_order():
            real_order.append([j + 1 for j in i])
        return [list(self.df.columns[i]) for i in real_order]

    def split_field(self):
        """
        return the fields to enable field-aware random split
        """
        return self.times.tolist()

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, index):
        return self.records[index], self.target[index]


class WSTDatasetWithTopology(QoSDataset):
    """
      The dataset also considers topology of as besides response-time, throughput, user ID, service ID
    """

    def __init__(self, dataset_path, data_task='RT'):
        """ read and process the dataset with pandas"""
        print("WSTDatasetWithTopology...")
        self.df = pd.DataFrame(pd.read_csv(dataset_path))
        self.data_task = data_task.lower()
        self.target = self.df["RT"].astype(np.float32)
        self.times = self.df["Slice_ID"].astype(np.float32)
        self.users = self.df["User_ID"].astype(np.int_)
        self.items = self.df["Service_ID"].astype(np.int_)
        self.num_users = np.max(self.users) + 1
        self.num_items = np.max(self.items) + 1
        self.items = self.items + self.num_users
        offset = self.num_users + self.num_items
        self.paths = self.df["ASPath"].apply(lambda x: x.split('-'))
        self.new_paths = []
        self.path_lens = []
        print("Re-assign identifier for path nodes...")
        self.as_set = set()
        for i in range(len(self.paths)):
            self.as_set.update(self.paths[i])
            path_i = [int(as_id) + offset for as_id in self.paths[i]]
            path_i.insert(0, self.users[i])
            path_i.append(self.items[i])
            self.new_paths.append(path_i)
            self.path_lens.append(len(path_i))
        # print(self.new_paths)
        print("Pad node sequences of paths...")
        self.paths = list(map(lambda x: torch.tensor(x), self.new_paths))
        self.paths = pad_sequence(self.paths, batch_first=True)
        # print(self.paths)

        """stack the arrays to form the full one """
        self.records = torch.column_stack((torch.tensor(self.times),
                                           torch.tensor(self.users),
                                           torch.tensor(self.items),
                                           self.paths,
                                           torch.tensor(self.path_lens)))

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, index):
        return self.records[index], self.target[index]

    def field_dims(self):
        """
        calculate the field dimensions, a.k.a, the total number of field objects
        """
        return [np.max(self.times) + 1, self.num_users, self.num_items, len(self.as_set)]

    def field_order(self):
        """
        in default case, list[0][...] is for time_slices, list[1][...] is for users, list[2][...]  is for items, list[3][...]  is for paths
        """
        return [[0], [1], [2], [3]]

    def dataset_fields(self):
        """
        return readable field names
        """
        real_order = []
        for i in self.field_order():
            real_order.append([j + 1 for j in i])
        return [list(self.df.columns[i]) for i in real_order]

    def split_field(self):
        """
        return the fields to enable field-aware random split
        """
        return self.times.tolist()


if __name__ == '__main__':
    dataset = WSTDatasetWithoutContext("../dataset/ws-time/zhou/rt_ctx.csv", 'RT')
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    print(dataset.split_field())

    dataset = WSTDatasetWithContext("../dataset/ws-time/zhou/rt_ctx.csv", 'RT')
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    print(dataset.split_field())

    dataset = WSTDatasetWithTopology("../dataset/ws-time/zhou/rt_topo.csv", 'RT')
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    print(dataset.split_field())
