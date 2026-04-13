import argparse
import gc
import logging
import os

import numpy as np
import torch
import yaml
from torch import nn, optim, sparse_coo_tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datatool.wsdt_dataset import WSDTDatasetWithoutContext
from datatool.wst_dataset import WSTDatasetWithContext, WSTDatasetWithoutContext, WSTDatasetWithTopology
from model.dynamic.memory_based.tuipcc import HybridForecast
from model.dynamic.memory_based.trcf import TRCFModel
from model.dynamic.model_based.tf.wspred import WSPredModel
from model.dynamic.model_based.mf.rncf import RecurrentNeuralCollaborativeFiltering
from model.dynamic.model_based.tf.deeptsqp import DeepTemporalAwareServiceQoSPrediction
from model.dynamic.model_based.tf.scatsf import SpatialContextAwareTimeSeriesForecast
from model.dynamic.model_based.mf.plmf import PersonalizedLSTMBasedMatrixFactorization
from model.dynamic.model_based.tf.costco import CoSTCoModel
from model.dynamic.model_based.tf.ntf import NeuralTensorFactorization
from model.dynamic.model_based.tf.gm import GraphModeling
from model.dynamic.model_based.tf.stf import SteadyandTransientFactorization
from util.eval import Evaluator
from util.seed import set_seed_for_all
from util.split import yet_another_random_split
from util.swats import SWATS


class Runner:
    def __init__(self, _args, _config):
        set_seed_for_all(_args.seed)
        self.num_workers = 0
        self.device = torch.device('cuda:' + _args.cuda if torch.cuda.is_available() else "cpu")
        self.dataset_name = _args.dataset_name
        self.dataset_path = _args.dataset_path
        self.data_task = _args.data_task
        self.loss_type = _args.loss_type
        self.batch_size = _args.batch_size
        self.batch_fold = _args.batch_fold
        self.weight_decay = _args.weight_decay
        self.epoch = _args.epoch
        self.learn_rate = _args.learn_rate
        self.contextual = _args.contextual
        self.embed_dim = _args.embed_dim
        self.cl_reg = _args.cl_reg
        self.dataset = self.get_dataset()
        self.field_dims = self.dataset.field_dims()
        self.field_order = self.dataset.field_order()
        self.criterion = None
        self.optimizer = None
        self.test_data = None
        self.train_data = None
        self.config = _config
        self.config['device'] = self.device
        self.config['embed_dim'] = _args.embed_dim
        self.config['rnn'] = _args.rnn

    def __str__(self):
        return (f"num_workers: {self.num_workers}\ndevice: {self.device}\ndataset_name: {self.dataset_name}\n"
                f"dataset_path: {self.dataset_path}\ndata_task: {self.data_task}\nembed_dim: {self.embed_dim}\n"
                f"loss_type: {self.loss_type}\nbatch_size: {self.batch_size}\nepoch: {self.epoch}\n"
                f"learn_rate: {self.learn_rate}\ncontextual: {self.contextual}\noptimizer: {self.optimizer}")

    def set_logger(self, logfile):
        self.logger = logging.getLogger("SparseNM")
        self.logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(logfile)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def get_dataset(self):
        match self.dataset_name.lower():
            case 'ctx':
                return WSTDatasetWithContext(self.dataset_path) if self.contextual else WSTDatasetWithoutContext(
                    self.dataset_path)
            case 'topo':
                return WSTDatasetWithTopology(self.dataset_path) if self.contextual else WSTDatasetWithoutContext(
                    self.dataset_path)
            case 'wsdt':
                return WSDTDatasetWithoutContext(self.dataset_path)
            case _:
                raise ValueError('unknown dataset name: ' + self.dataset_name)

    def set_model(self, model_name):
        self.logger.info("MODEL: " + model_name.upper())
        match model_name.lower():
            case 'tuipcc':
                self.model = HybridForecast(self.train_data, self.field_dims, self.config)
            case 'wspred':
                self.model = WSPredModel(self.train_data, self.field_dims, self.field_order, self.config)
            case 'gm':
                self.model = GraphModeling(self.train_data, self.field_dims, self.field_order, self.config)
            case 'costco':
                self.model = CoSTCoModel(self.field_dims, self.field_order, self.config)
            case 'plmf':
                self.model = PersonalizedLSTMBasedMatrixFactorization(self.field_dims, self.field_order, self.config)
            case 'rncf':
                self.model = RecurrentNeuralCollaborativeFiltering(self.field_dims, self.field_order, self.config)
            case 'deeptsqp':
                self.model = DeepTemporalAwareServiceQoSPrediction(self.train_data, self.field_dims, self.field_order,self.config)
            case 'scatsf':
                self.model = SpatialContextAwareTimeSeriesForecast(self.train_data, self.test_data, self.field_dims,self.field_order,self.config)
            case 'ntf':
                self.model = NeuralTensorFactorization(self.field_dims, self.field_order, self.config)
            case 'trcf':
                self.model = TRCFModel(self.train_data, self.field_dims, self.field_order, self.config)
            case 'stf':
                self.model = SteadyandTransientFactorization(self.train_data, self.field_dims,self.field_order,self.config)
            case _:
                raise ValueError('unknown model name: ' + model_name)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"#Parameters: {total / 1e6:.2f} million\t #Memory:{total * 4 / 1e6:.2f} M")
        self.model.to(self.device)
        print(self.model)

    def split_dataset(self, _density):
        self.train_data, self.test_data = yet_another_random_split(self.dataset, [_density, 1 - _density])
        return (DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True),
                DataLoader(self.test_data, batch_size=self.batch_fold * self.batch_size, num_workers=self.num_workers,
                           shuffle=False))

    def train(self, data_loader):
        """
        :param data_loader: The training data loader
        :return: The averaged loss over all mini-batches
        """
        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        total_loss = 0.0
        if isinstance(self.model, WSPredModel):
            self.num_times = self.field_dims[0]
            self.num_users = self.field_dims[1]
            self.num_items = self.field_dims[2]

            index = self.train_data[:][0]
            value = np.array(self.train_data[:][1])

            sparse_r, sparse_c = [None] * self.num_times, [None] * self.num_times

            for i in range(self.num_times):
                ind = np.squeeze(np.argwhere(index[:, 0] == i))
                sub_index = index[ind][:, [1, 2]]
                sub_value = value[ind]

                sparse_r[i] = sparse_coo_tensor(
                    sub_index.T, sub_value.T, [self.num_users, self.num_items]
                )
                sparse_c[i] = sparse_coo_tensor(
                    sub_index.T, np.ones(sub_value.shape[0]), [self.num_users, self.num_items]
                )

            for i in range(1, self.num_times):
                sparse_r[0] = sparse_r[0].add(sparse_r[i])
                sparse_c[0] = sparse_c[0].add(sparse_c[i])

            spr = sparse_r[0].coalesce()
            spc = sparse_c[0].coalesce()

            self.y_us = sparse_coo_tensor(
                spr.indices(),
                spr.values() / spc.values().detach(),
                [self.num_users, self.num_items]
            ).to_dense().to(self.device)


        for x, y in tqdm(data_loader, smoothing=0, mininterval=1.0):
            x, y = x.to(self.device), y.to(self.device)
            if isinstance(self.model, WSPredModel):
                offset = x[:, 1] * self.num_items + x[:, 2]
                y2 = torch.take(self.y_us.view(-1), offset)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y.float())
                loss += self.config['WSPred']['eta'] * self.criterion(y_hat, y2)
                total_loss += loss.item()
            if isinstance(self.model, GraphModeling):
                y_hat = self.model(x, 'train')
                loss = self.criterion(y_hat, y.float())
            else:
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y.float())

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def test(self, data_loader):
        """
        :param data_loader: The test data loader
        :return: y_pred, y_true
        """
        self.model.eval()
        y_list, y_hat_list = list(), list()
        with torch.no_grad():
            for x, y in tqdm(data_loader, smoothing=0, mininterval=1.0):
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                y_list.extend(y.tolist())
                y_hat_list.extend(y_hat.tolist())
        return y_hat_list, y_list

    def set_criterion(self, loss_type):
        criterion_map = {
            'mse': nn.MSELoss(),
            'l2': nn.MSELoss(),
            'huber': nn.HuberLoss(),
            'mae': nn.L1Loss(),
            'l1': nn.L1Loss(),
            'smoothl1': nn.SmoothL1Loss()
        }
        self.criterion = criterion_map.get(loss_type.lower(), nn.MSELoss())

    def set_optimizer(self, optimizer_name='Adam'):
        optimizer_map = {
            'adam': optim.Adam(params=self.model.parameters(), lr=self.learn_rate, weight_decay=self.weight_decay),
            'sgd': optim.SGD(params=self.model.parameters(), lr=self.learn_rate, weight_decay=self.weight_decay),
            'adagrad': optim.Adagrad(params=self.model.parameters(), lr=self.learn_rate,
                                     weight_decay=self.weight_decay),
            'swats': SWATS(params=self.model.parameters(), lr=self.learn_rate, weight_decay=self.weight_decay)
        }
        self.optimizer = optimizer_map.get(optimizer_name.lower(),
                                           optim.Adam(params=self.model.parameters(), lr=self.learn_rate,
                                                      weight_decay=self.weight_decay))


if __name__ == '__main__':
    with open("./config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='wsdt',
                        help='ctx for rt_ctx.csv, topo for rt_topo.csv, wsdt for rtdata.txt/tpdata.txt')
    parser.add_argument('--dataset_path', type=str, default='dataset/ws-time/wsdream/rtdata.txt',
                        help='the path of dataset')
    parser.add_argument('--data_task', type=str, default='RT', help='the prediction task')
    parser.add_argument('--loss_type', type=str, default='L1', help='the loss type of model')
    parser.add_argument('--batch_size', type=int, default=1024, help='the size of mini_batch')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay for optimizer')
    parser.add_argument('--epoch', type=int, default=50, help='the training epochs')
    parser.add_argument('--learn_rate', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--cl_reg', type=float, default=2e-1, help='the weight for contrastive loss')
    parser.add_argument('--model_name', type=str, default='gcn', help='the prediction model')
    parser.add_argument('--rnn', type=str, default='gru|lstm|tkan', help='the rnn encoder')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--contextual', type=bool, default=False, help='disable contextual features')
    parser.add_argument('--seed', type=int, default=3305)
    parser.add_argument('--cuda', type=str, default='0', help='the gpu to use')
    parser.add_argument('--batch_fold', type=int, default=1, help='to accelerate inference time')
    parser.add_argument('--logpath', type=str, default='./log/dynamic/', help='the log path')
    parser.add_argument('--embed_dim', type=int, default='16', help='the embedding dims of input features')

    cf_models = ( HybridForecast,TRCFModel)
    args = parser.parse_args()
    runner = Runner(args, config)
    runner.set_logger(f"{args.logpath}/{args.data_task}-{args.model_name}-{args.embed_dim}.log")
    eva = Evaluator(['MAE', 'NMAE', 'RMSE', 'MAPE'], runner.epoch)
    for density in [0.01,0.02,0.04,0.06,0.08]:
        train_dataloader, test_dataloader = runner.split_dataset(density)
        runner.set_model(args.model_name)
        if not isinstance(runner.model, cf_models):
            runner.set_criterion(args.loss_type)
            runner.set_optimizer(args.optimizer)
            scheduler = torch.optim.lr_scheduler.StepLR(runner.optimizer, step_size=10, gamma=0.5)
        runner.logger.info(str(runner))
        for epoch_i in range(runner.epoch):
            if not isinstance(runner.model, cf_models):
                loss = runner.train(train_dataloader)
                runner.logger.info(f'epoch:{epoch_i:3d} train loss:{loss:.8f}')
                scheduler.step()
                if(epoch_i % 10 ==0):
                    lr = runner.optimizer.param_groups[0]['lr']
                    runner.logger.info(f'epoch:{epoch_i:3d} current lr: {lr:.6f}')
            if (epoch_i >= runner.epoch - 10):
                y_pred, y_true = runner.test(test_dataloader)
                eva.update(y_pred, y_true)
                if hasattr(runner.model, 'params'):
                    params = runner.model.params.detach().cpu()
                    params_list = [round(v.item(), 6) for v in params]
                    runner.logger.info(f'epoch:{epoch_i:3d} params: {params_list}')
                runner.logger.info(f'epoch:{epoch_i:3d} test {eva.format_compute(eva.compute())}')
        runner.logger.critical(f'BEST: {args.model_name.upper()} SEED={args.seed} MD={density} {eva.best_metric()}')
        gc.collect()