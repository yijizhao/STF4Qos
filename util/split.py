"""
@author Hao Wu
@email haowu@ynu.edu.cn
@date  July 27, 2023
@version 0.1

"""
from torch import randperm
from torch.utils.data import Subset
from tqdm import tqdm

from datatool.ws_dataset import WSDatasetWithContext


def _accumulate(iterable, fn=lambda x, y: x + y):
    'Return running totals'
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def yet_another_random_split(dataset, ratio):
    r"""
    Randomly split a dataset into non-overlapping new datasets where assign the records in dataset to different bucket units.

      param ratio; [0.05,0.05,0.90]; [0.05, 0.95]
      param dataset (Dataset): Dataset to be split
    """
    field = dataset.split_field()

    bucket = [[] for i in range(len(field))]
    for idx in tqdm(range(len(field))):
        bucket[field[idx]].append(idx)

    global_indices = [[] for i in range(len(ratio))]
    for i in tqdm(range(len(bucket))):
        if len(bucket[i]) < 2: continue
        lengths = []
        for j in range(len(ratio) - 1):
            lengths.append(int(len(bucket[i]) * ratio[j]))
        lengths.append(len(bucket[i]) - sum(lengths))
        local_indices = randperm(sum(lengths)).tolist()
        k = 0
        for offset, length in zip(_accumulate(lengths), lengths):
            global_indices[k].extend([bucket[i][l] for l in local_indices[offset - length:offset]])
            k += 1

    return [Subset(dataset, global_indices[i]) for i in range(len(global_indices))]


if __name__ == '__main__':
    dataset = WSDatasetWithContext("../dataset/ws-topo/rt_tp_ctx.csv", 'RT')
    print(dataset.dataset_fields())
    print(dataset.field_dims())
    print(dataset.field_order())
    train_data, test_data = yet_another_random_split(dataset, [0.05, 0.95])
    print(train_data[:][0][:2].T)
