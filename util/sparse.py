import numpy as np
import torch
from torch import nn


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse_coo_tensor(indices, values, size)


def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs = emb[cols] * torch.unsqueeze(sp.values(), dim=1)
    result = torch.zeros((sp.shape[0], emb.shape[1])).cuda(device)
    result.index_add_(0, rows, col_segs)
    return result
