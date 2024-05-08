from pathlib import Path
import pickle
from typing import Dict, Tuple, List
import numpy as np
import tqdm
import torch
from torch import nn

DATA_PATH = "../data/"


class Train(object):
    def __init__(
            self, model, name, rel_size,
            optimizer, batch_size: int = 256,
            verbose: bool = True, is_cuda: bool = True
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

    def epoch(self, examples: torch.LongTensor, args, mode, neis_all, neis_timestamps, path_weight, epoch):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # 随机打乱
        criterion = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[b_begin:b_begin + self.batch_size].to('cuda' if self.is_cuda else 'cpu')
                # 分支 if rel > rel_size
                batch_his = []  # (batch_size , his不等长的[]列表)
                batch_time = []  # (batch_size ,1)
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
                        batch_time.append(time.item())  # current time

                # batch_time = torch.tensor(batch_time).to(args.cuda)
                predictions, contrastive_leanring_loss, time_ = self.model.forward(
                    args, input_batch, batch_his, mode, neis_all, neis_timestamps, path_weight, epoch)  # 预测 batch
                l_time = self.learn_time(time_)

                truth = input_batch[:, 2]
                l_link_predic = criterion(predictions, truth)  # 预测损失
                # loss = l_link_predic + 0.01 * contrastive_leanring_loss + 0.1 * l_time #good 56.01	44.83	63.64	76.02
                loss = l_link_predic + 0.1 * contrastive_leanring_loss + 0.1 * l_time  # good  0.5704	0.4544	0.6503	0.7717
                # loss = l_link_predic + 0.2 * contrastive_leanring_loss + 0.5 * l_time
                # loss = l_link_predic + 0.1 * contrastive_leanring_loss + 0. * l_time
                # loss = l_link_predic + l_time
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    link_loss=f'{l_link_predic.item():.4f}',
                    time_loss=f'{l_time.item():.4f}',
                    loss_c_l=f'{contrastive_leanring_loss.item():.4f}'
                )

    def learn_time(self, time):
        l = 0
        for f in time:
            rank = int(f.shape[1] / 2)
            ddiff = f[1:] - f[:-1]
            diff = torch.sqrt((ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2)) ** 4
            l = l + torch.sum(diff)
        return l / time[0].shape[0]

    def learn_embedding(self, embed):
        norm = 0
        if embed is not None:
            for f in embed[:3]:
                norm += self.weight * torch.sum(torch.abs(f) ** 3)
            for f in embed[3:]:
                norm += self.weight * torch.sum(torch.abs(f) ** 3)
            return norm / embed[0].shape[0]


class Dataset(object):
    def __init__(self, name: str, is_cuda: bool = False):
        self.root = Path(DATA_PATH) / name  # "/"操作符表示路径的连接
        self.is_cuda = is_cuda
        self.data = {}  # all data
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)  # 二维数组，计算所有每列的最大值，得到[max_col1, max_col2, max_col3, max_col4]
        maxis2 = np.max(self.data['test'], axis=0)
        maxis3 = np.max(self.data['valid'], axis=0)
        # 计算最大值，是为了在models只给你用于对实体与关系nn.embedding(,)初始化
        self.n_entities = int(
            max(max(max(maxis[0], maxis[2]), max(maxis2[0], maxis2[2])), max(maxis3[0], maxis3[2])) + 1)
        self.n_predicates = int(max(max(maxis[1], maxis2[1]), maxis3[1]) + 1)  # 最大的关系编号，是需要 +1 的
        self.n_predicates *= 2
        self.n_timestamps = int(max(max(maxis[3], maxis2[3]), maxis3[3]) + 1)  # 最大的时间编号
        self.n_timestamps = int(max(max(maxis[3], maxis2[3]), maxis3[3]) + 1)  # 最大的时间编号
        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')  # 读取正向与逆向history
        self.to_skip: Dict[str, Dict[Tuple[int, int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def has_intervals(self):
        return self.events is not None

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        origin_data = np.vstack((self.data['train'], copy))  # 列堆叠  ,（o,rel>=230,s,t）第一个已经是o了
        return origin_data

    def eval(self, model, split: str, args, mode, neis_all, neis_timestamps, path_weight, epoch):
        # if self.events is not None:
        #     return self.time_eval(model, split, n_queries, 'rhs', at)
        test = self.data[split]  # get the valid/test dataset
        examples = torch.from_numpy(test.astype('int64')).to('cuda' if self.is_cuda else 'cpu')
        missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        mean_rank = {}
        hits_at = {}
        # added = {'lhs':{}, 'rhs':{}}

        for m in missing:
            q = examples.clone()
            ranks = torch.ones(len(q))
            _his = []  # (batch_size , his不等长列表)
            # _time = [] # (batch_size ,1)
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            batch_size: int = 500
            b_begin = 0
            while b_begin < examples.shape[0]:
                batch_input = q[b_begin:b_begin + batch_size]
                # batch_time = _time[b_begin:b_begin + batch_size]
                # batch_his = _his[b_begin:b_begin + batch_size]
                scores, targets = model.forward(args, batch_input, _his, mode, neis_all, neis_timestamps, path_weight,
                                                epoch, self.to_skip[m])
                ranks[b_begin:b_begin + batch_size] += torch.sum((scores >= targets).float(), dim=1).cpu()
                b_begin += batch_size

            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))
            # added[m]['q'] = q
            # added[m]['rank'] = ranks
        return mean_rank, mean_reciprocal_rank, hits_at  # , added

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities, self.n_timestamps

