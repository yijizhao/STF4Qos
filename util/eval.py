"""
@author Hào Wu
@email haowu@ynu.edu.cn
@date  July 24, 2023
@version 0.1

"""
import torch


class Evaluator(object):
    """
    A pytorch implementation of various evaluation metrics w.r.t regression tasks.
    """

    def __init__(self, metrics, epoch):
        super().__init__()
        self.y_true = torch.zeros(0)
        self.y_pred = torch.zeros(0)
        self.metrics = metrics
        self.cache = []
        self.epoch = epoch

    def update(self, y_pred, y_true, ):
        self.y_pred = torch.Tensor(y_pred)
        self.y_true = torch.Tensor(y_true)

    def mean_absolute_error(self):
        return torch.mean(torch.abs(self.y_pred - self.y_true))

    def normalized_mean_absolute_error(self):
        return self.mean_absolute_error() / (torch.max(self.y_true) - torch.min(self.y_true))

    def mean_squared_error(self):
        return torch.mean(torch.square(self.y_pred - self.y_true))

    def root_mean_squared_error(self):
        return torch.sqrt(self.mean_squared_error())

    def mean_absolute_percentage_error(self):
        return torch.mean(torch.abs((self.y_pred - self.y_true) / self.y_true)) * 100

    def best_metric(self, criterion='MAE', fmt='.4f'):
        t = torch.min(torch.tensor(self.cache), 0)
        val = t.values.tolist()
        ind = t.indices.tolist()
        sf1 = [self.metrics[i] + ":" + format(val[i], fmt) + "@" + format(ind[i], '<3d') for i in
               range(len(self.metrics))]
        select_epoch = ind[self.metrics.index(criterion)]
        select_vals = self.cache[select_epoch]
        sf2 = '[' + ' '.join([format(v, fmt) for v in select_vals]) + "]@" + format(select_epoch, '<3d')
        return ' '.join(sf1) + ' ' + sf2

    def compute(self):
        error = []
        if len(self.cache) == self.epoch:
            self.cache.clear()
        for m in self.metrics:
            error.append(float(self.get_metric(m)))
        self.cache.append(error)
        return error

    def get_metric(self, metric):
        match metric:
            case 'MAE':
                return self.mean_absolute_error()
            case 'NMAE':
                return self.normalized_mean_absolute_error()
            case 'MSE':
                return self.mean_squared_error()
            case 'RMSE':
                return self.root_mean_squared_error()
            case 'MAPE':
                return self.mean_absolute_percentage_error()

    def format_compute(self, error, fmt='.4f'):
        sf = [self.metrics[i] + ":" + format(error[i], fmt) for i in range(len(error))]
        return ' '.join(sf)
