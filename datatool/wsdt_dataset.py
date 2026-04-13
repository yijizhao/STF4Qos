"""
@author Hao Wu
@email haowu@ynu.edu.cn
@date  July 17, 2023
@version 0.1

"""
from sys import getsizeof

import numpy as np
import pandas as pd

from datatool.qos_dataset import QoSDataset




class WSDTDatasetWithoutContext(QoSDataset):
    """
      The dataset  considers only the fields of response-time/throughput, slice ID, user ID, service ID
    """

    def __init__(self, dataset_path):
        """ read and process the dataset with pandas"""
        print("WSDTDatasetWithoutContext...")
        self.df = pd.DataFrame(pd.read_csv(dataset_path, header=None))
        self.target = self.df[3].astype(np.float32)
        self.times = self.df[2].astype(np.int_)
        self.items = self.df[1].astype(np.int_)
        self.users = self.df[0].astype(np.int_)
        #print("max user_id:", np.max(self.users))
        #print("max item_id:", np.max(self.items))

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


if __name__ == '__main__':
    dataset = WSDTDatasetWithoutContext("../dataset/ws-time/wsdream/tpdata.txt")
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    print(len(dataset))

    dataset = WSDTDatasetWithoutContext("../dataset/ws-time/wsdream/rtdata.txt")
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    print(len(dataset))
