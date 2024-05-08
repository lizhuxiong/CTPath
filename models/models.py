from typing import Tuple, List, Dict
from neibs_info_embedding import Path
from contrastive_learning import ConLoss_his
from his_embdding import EventEmbeddingModel
import torch
from torch import nn
import torch.nn.functional as F


class Supercomplex(nn.Module):
    def __init__(
            self, args, sizes: Tuple[int, int, int, int], rank: int,
            is_cuda: bool = False
    ):
        super(Supercomplex, self).__init__()
        self.model_name = "Supercomplex"
        self.sizes = sizes
        self.rank = rank  # dimension
        # self.pi = 3.14159265358979323846
        # self.lam = 1
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),  # 0 entityies (7128,1200*2)
            nn.Embedding(sizes[1], rank, sparse=True),  # 1 relaitions
            nn.Embedding(sizes[3], rank, sparse=True),  # 2 表示关系的timstamps
            nn.Embedding(sizes[3], rank, sparse=True),  # 3 表示实体的timstamps
            nn.Embedding(sizes[3], rank, sparse=True),  # 4 entity timstamps1
            # nn.Embedding(sizes[3], rank, sparse=True),  # 5
            nn.Embedding(sizes[0], rank, sparse=True),  # 6 历史信息
            # nn.Embedding(sizes[1], rank, sparse=True),  # 6 relation weight

        ])
        self.is_cuda = is_cuda

        feature_length = rank
        self.pathNet_encoder = Path(feature_length, args).to(args.cuda)
        # self.contrastive_leanring_path = ConLoss_Path(args).to(args.cuda)
        self.contrastive_leanring_his = ConLoss_his(args).to(args.cuda)

        self.his_emb_model = EventEmbeddingModel(args.cuda, rank, rank)

        if rank % 2 != 0:
            raise "rank need to be devided by 2.."
        for i in range(len(self.embeddings)):
            torch.nn.init.xavier_uniform_(self.embeddings[i].weight.data)

    @staticmethod
    def has_time():
        return False

    def mul(self, emb1, emb2):
        a, b = torch.chunk(emb1, 2, dim=1)
        c, d = torch.chunk(emb2, 2, dim=1)
        return torch.cat([(a * c - b * d), (a * d + b * c)], dim=1)

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def complex_mul(self, emb1, emb2):
        a, b = torch.chunk(emb1, 2, dim=1)
        c, d = torch.chunk(emb2, 2, dim=1)
        return torch.cat(((a * c + b * d), (a * d - b * c)), dim=1)  # 沿列拼接  #复数公式计算错误？ 没有，其实不是复数计算，就是元素乘积

    def forward(self, args, x, batch_history, mode, neis_all, neis_timestamps, path_weight, epoch,
                filters: Dict[Tuple[int, int, int], List[int]] = None):
        # get the his_emb

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])

        all_en = self.embeddings[0].weight.transpose(0, 1)
        en_his = self.embeddings[5](x[:, 0])

        time_ent = self.embeddings[2](x[:, 3])
        # time_ent2 = self.embeddings[4](x[:, 3])
        time_rel = self.embeddings[3](x[:, 3])
        # time_rel2 = self.embeddings[5](x[:, 3])
        # rel_ = self.mul(time_rel2, rel)
        # # rel_weight_ = self.mul(rel_weight, comp_time)
        # rel = rel + rel_ #+ torch.sin(time_rel)# + rel_weight_

        neis_embd = self.forward_PathNet(self.pathNet_encoder, self.embeddings[0].weight, self.embeddings[2].weight,
                                         args.num_walks, args.walk_len,
                                         neis_all, neis_timestamps, path_weight, x[:, 0], x[:, 3])

        # en_his = self.mul(neis_embd, en_his)
        # lhs2 = neis_embd + self.mul(time_ent2, en_his)
        # lhs = lhs + en_his + lhs2 #+ torch.sin(time_ent) #+ en_weight   # 不要周期时间
        # # lhs = lhs1 + torch.sin(time_ent)# + en_weight

        # 1)
        en_his_ = self.mul(neis_embd, en_his)  # neis_embd作为path信息，en_his作为其中的一个作用力
        lhs2 = neis_embd + self.mul(time_ent, en_his_) #time_ent2可以换为time_ent
        lhs4 = self.mul(time_ent, lhs)
        lhs_ = lhs + lhs2 + lhs4


        # en_his_ = neis_embd + self.mul(neis_embd, lhs)
        # lhs4 = self.mul(time_ent, lhs)
        # lhs_ = lhs + en_his_ + lhs4

        # en_his_ = self.mul(neis_embd, en_his)
        # lhs2 = en_his + self.mul(time_ent2, en_his_)
        # lhs4 = self.mul(time_ent, lhs)
        # # lhs = lhs + lhs2 + lhs3 + lhs4
        # lhs_ = lhs + lhs2 + lhs4

        # 2) no time
        # en_his_ = self.mul(neis_embd, en_his)  # neis_embd作为path信息，en_his作为其中的一个作用力
        # lhs2 = neis_embd + self.mul(time_ent, en_his_) #time_ent2可以换为time_ent
        # lhs_ = lhs + lhs2

        # 3) the path no time
        # en_his = neis_embd + self.mul(neis_embd, en_his)
        # # lhs2 = neis_embd + self.mul(time_ent2, en_his)
        # # lhs3 = self.mul(rel, lhs)
        # lhs4 = self.mul(time_ent, lhs)
        # lhs = lhs + en_his + lhs4

        # 4) no path
        # lhs4 = self.mul(time_ent, lhs)
        # lhs_ = lhs + lhs4

        # 5) no all

        if mode == 'Training':
            # get the contrastive_leanring_loss
            # contrastive_leanring_loss_path = self.contrastive_leanring_path(x[:, 0], self.embeddings[0].weight, neis_path_2)
            # curen_embedding = {}
            # for cur_en, embedding in zip(x[:, 0], lhs):
            #     curen_embedding[cur_en] = embedding

            # contrastive_leanring_loss = self.contrastive_leanring_his(x[:, 0], curen_embedding, self.embeddings[0].weight, time_ent, batch_history)
            # contrastive_leanring_loss = self.contrastive_leanring_his(x[:, 0], lhs2, self.embeddings[5].weight, time_ent2, batch_history) # path
            # contrastive_leanring_loss = self.contrastive_leanring_his(x[:, 0], lhs, self.embeddings[5].weight, time_ent, batch_history) # entity
            contrastive_leanring_loss = self.contrastive_leanring_his(x[:, 1], self.embeddings[1].weight, time_rel, batch_history)  # rel
            lhs_rel = self.mul(torch.cat([rel, time_rel], dim=1),
                               torch.cat([lhs_, time_ent], dim=1))

            # lhs_rel = F.dropout(lhs_rel, p=0.1, training=self.training)
            a, b = torch.chunk(lhs_rel, chunks=2, dim=1)
            # c, d= torch.chunk(lhs_rel2, chunks=2, dim=1)
            # scores = a @ all_en + torch.sum(b * neis_embd, dim=1).unsqueeze(1) #+ torch.sum(a*lhs,dim=1).unsqueeze(1) + torch.sum(b*time_ent,dim=1).unsqueeze(1)
            # scores = a @ all_en + torch.sum(b*lhs_,dim=1).unsqueeze(1)
            # scores = a @ all_en + torch.sum(b * lhs2, dim=1).unsqueeze(1)# + torch.sum(b*time_ent,dim=1).unsqueeze(1)
            # scores = b @ all_en + torch.sum(a * lhs2, dim=1).unsqueeze(1)# + torch.sum(b*time_ent,dim=1).unsqueeze(1)
            scores = a @ all_en + torch.sum(b * neis_embd, dim=1).unsqueeze(1)  # best
            # scores = a @ all_en + b @ all_en # firtst epoch maximum
            return scores, contrastive_leanring_loss, (
                self.embeddings[2].weight[:-1],
                self.embeddings[3].weight[:-1],
                self.embeddings[4].weight[:-1]
            )

        else:
            with torch.no_grad():
                lhs_rel = self.mul(torch.cat([rel, time_rel], dim=1),
                                   torch.cat([lhs_, time_ent], dim=1))  # to obtain 𝑄(𝑠, 𝑝, 𝜏) 得到相互作用*****2*****
                a, b = torch.chunk(lhs_rel, chunks=2, dim=1)
                # c, d= torch.chunk(lhs_rel2, chunks=2, dim=1)
                # scores = a @ all_en + torch.sum(b * neis_embd, dim=1).unsqueeze(1) #+ torch.sum(a*lhs,dim=1).unsqueeze(1) + torch.sum(b*time_ent,dim=1).unsqueeze(1)
                # scores = a @ all_en + torch.sum(b*lhs_,dim=1).unsqueeze(1)
                # scores = a @ all_en + torch.sum(b * lhs2, dim=1).unsqueeze(1)# + torch.sum(b*time_ent,dim=1).unsqueeze(1)
                # scores = b @ all_en + torch.sum(a * lhs2, dim=1).unsqueeze(1)# + torch.sum(b*time_ent,dim=1).unsqueeze(1)
                scores = a @ all_en + torch.sum(b * neis_embd, dim=1).unsqueeze(1)  # best
                # scores = a @ all_en + b @ all_en # firtst epoch maximum
                targets = []
                for (score, target_index) in zip(scores, x[:, 2]):
                    targets.append(score[target_index])
                targets = torch.tensor(targets).view(-1, 1).to(args.cuda)
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, query in enumerate(x):
                    filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]  # 不预测同步记录
                    filter_out += [x[i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6

            # if (mode == 'test' and epoch >=6):
            #     flag = 0
            #     for i in range(scores.shape[0]):
            #         if targets[i] >= torch.max(scores[i]):
            #             flag =flag+1
            #             values, indices = torch.topk(scores[i], k=3)
            #             top1_score = values[0].item()
            #             top1_en = indices[0].item()
            #             top2_score = values[1].item()
            #             top2_en = indices[1].item()
            #             top3_score = values[2].item()
            #             top3_en = indices[2].item()
            #             # print('top1_score:',top1_score,
            #             #       ' top1_en:',top1_en,
            #             #       ' top2_score:', top2_score,
            #             #       ' top2_en:', top2_en,
            #             #       ' top3_score:', top3_score,
            #             #       ' top3_en:', top3_en)
            #             if flag == 10:
            #                 break
            return scores, targets

    def forward_PathNet(self, pathNet, entity_embedding, en_time_embedding, num_walks, walk_len,
                        neis_all, neis_timestamps, path_weight,
                        entitys, cur_time):
        neis = neis_all[entitys]
        neis_time = neis_timestamps[entitys]
        weight = path_weight[entitys]

        for index, time in enumerate(neis_time):
            if time[:1].item() == -1:  # 处理没有path的节点  [13, 13, 13, -1, -1, -1, 0, 0, 0]  -> [13, 13, 13, en_time, en_time, en_time, 1, 1, 1]
                neis_time[index].fill_(cur_time[index])
                # weight[index].fill_(1)
        # pathNet.train()
        # entity_embedding shape(7129, 200)
        neibs_info_embedding = pathNet(entity_embedding, en_time_embedding, neis, num_walks, walk_len, neis_time,
                                       weight)
        return neibs_info_embedding  # shape(the number of all nodes , hidden_size) ==> shape(the number of current epoch, hidden_size)=(neis.shape[0], hidden_size)
