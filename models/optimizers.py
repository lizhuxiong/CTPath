import tqdm
import torch
from torch import nn
from torch import optim

# from models import TKBCModel
from regularizers import Regularizer
import pickle
from typing import Dict, Tuple, List
from pathlib import Path
DATA_PATH = "../data/"


class TKBCOptimizer(object):
    def __init__(
            self, model, name, rel_size,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True, add_reg=None, is_cuda: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.is_cuda = is_cuda

        self.rel_size = rel_size
        # print("------rel_size------: ", rel_size)
        self.his_direction = ['lhs', 'rhs']
        self.root = Path(DATA_PATH) / name
        his_f = open(str(self.root / f'history.pickle'), 'rb')  # 读取正向与逆向历史事件
        # self.history: Dict[str, Dict[Tuple[int, int, int], List[Tuple]]] = pickle.load(his_f)
        self.history: Dict[str, Dict[Tuple[int, int, int], List[List]]] = pickle.load(his_f)
        his_f.close()

    def epoch(self, examples: torch.LongTensor, args, mode, neis_all, path_distance, neis_path_2):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # 随机打乱
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[b_begin:b_begin + self.batch_size].to('cuda' if self.is_cuda else 'cpu')
                # 分支 if rel > rel_size 
                batch_his = []  # (batch_size , his不等长的[]列表)
                # batch_time = [] # (batch_size ,1)
                for en1, rel, en2, time in input_batch:
                    if rel.item() >= self.rel_size:  # 逆向预测
                        his = self.history[self.his_direction[0]][
                            (en1.item(), rel.item(), time.item())]  # en2这些仍是张量， en1已经对应逆向
                    else:  # 正向预测
                        his = self.history[self.his_direction[1]][(en1.item(), rel.item(), time.item())]
                    if (len(his) == 0):
                        batch_his.append([])
                    else:
                        batch_his.append(his)
                        # batch_time.append(time.item()) #current time

                # batch_time = torch.tensor(batch_time).to(args.cuda)
                predictions, contrastive_leanring_loss = self.model.forward(args, input_batch, batch_his, mode,
                                                                            neis_all, path_distance,
                                                                            neis_path_2)  # 预测 batch
                # predictions, factors, time, contrastive_leanring_loss = self.model.forward(args, input_batch, batch_time, batch_his, mode, neis_all, path_distance, neis_path_2)  #预测 batch
                # predictions, factors, time = self.model.forward(input_batch, 'Training')  #预测
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)  # 预测损失
                l = l_fit +contrastive_leanring_loss

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.4f}',
                    # clloss=f'{contrastive_leanring_loss.item():.4f}'
                )