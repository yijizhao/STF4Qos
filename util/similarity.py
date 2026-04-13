"""
@author Hao Wu
@email haowu@ynu.edu.cn
@date  July 19, 2023
@version 0.1

"""

import torch

def PearsonCorrleationCoefficent(tensor):
    _tensor = tensor.clone().detach()
    """ the average rating in term of rows"""
    mean = torch.sum(_tensor, dim=1) / torch.count_nonzero(_tensor, dim=1)
    _tensor = _tensor - mean.unsqueeze(-1)
    _tensor = _tensor.index_put(torch.nonzero(tensor <= 0, as_tuple=True), torch.tensor(0.))
    # print(_tensor.dtype)
    """ a fast trick to compute pairwise similarity  """
    _tensor = _tensor / torch.norm(_tensor, dim=-1, keepdim=True)  
    sim = torch.mm(_tensor, _tensor.T)  
    """  set the diag elements to zero to exclude the user itself """
    diag = torch.diag(sim)  
    sim = sim - torch.diag_embed(diag)  
    # to exclude NaN values
    sim = torch.where(torch.isnan(sim), torch.full_like(sim, 0), sim)
    mean = torch.where(torch.isnan(mean), torch.full_like(mean, 0), mean)
    return sim, mean


def CosineSimilarity(tensor):
    _tensor = tensor.clone().detach()
    """ a fast trick to compute pairwise similarity  """
    _tensor = _tensor / torch.norm(_tensor, dim=-1, keepdim=True)  
    sim = torch.mm(_tensor, _tensor.T)  
    """  set the diag elements to zero to exclude the user itself """
    diag = torch.diag(sim)  
    sim = sim - torch.diag_embed(diag)  
    sim = torch.where(torch.isnan(sim), torch.full_like(sim, 0), sim)
    return sim



if __name__ == "__main__":
    
    X = torch.tensor([
        [4, 3, 0, 5, 0],
        [0, 2, 4, 0, 1],
        [3, 0, 0, 4, 2],
        [0, 0, 0, 0, 0]  
    ], dtype=torch.float32)

    pcc, means = PearsonCorrleationCoefficent(X)

    print("非零均值:")
    print(means)
    print("\nPCC相似度矩阵:")
    print(pcc)

    cos = CosineSimilarity(X)
    print("\nCOS相似度矩阵:")
    print(cos)
