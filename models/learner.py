import argparse
from typing import Dict
import torch
from torch import optim
import codecs
import tqdm
from datasets import Dataset, Train
from models import *
import os
import time
import numpy as np

parser = argparse.ArgumentParser(description="Combine Time")
parser.add_argument('--dataset', type=str, default= 'ICEWS14',help="Dataset name")
parser.add_argument('--model', type=str, default= 'Supercomplex')
parser.add_argument('--max_epochs', default=20, type=int,help="Number of epochs.")
parser.add_argument('--valid_freq', default=1, type=int,help="Number of epochs between each valid.")
parser.add_argument('--rank', default=1000, type=int,help="Factorization rank.")
parser.add_argument('--batch_size', default=1000, type=int,help="Batch size.")
parser.add_argument('--learning_rate', default=0.1, type=float,help="Learning rate")
parser.add_argument('--gpu', default=1, type=int,help="Use CUDA for training")
parser.add_argument('--cuda', type=str, default='cuda:0')
parser.add_argument('--dropout_pathnet', type=float, default=0.2)
parser.add_argument("--n_hidden", type=int, default=160, help="number of pathnet hidden units")
parser.add_argument("--num_walks", type=int, default=1)
parser.add_argument("--walk_len", type=str, default=2)
parser.add_argument("--num_walks_neis", type=int, default=20)
parser.add_argument("--walk_len_neis", type=str, default=2)

args = parser.parse_args()



# def load_merw(args):
#     name = args.dataset
#     num_of_walks = args.num_walks
#     walk_length = args.walk_len
#     # num_walks_neis = args.num_walks_neis
#     # walk_len_neis = args.walk_len_neis
#
#     walks = []  # walks = {list: n} [[0, 44, 397, 266], [0, 74, 66, 196], [0, 160, 137, 358]]
#     # walks_2 = []
#     path_distance = []  # path_type = {list: n} [[0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2]]
#     path_weight = []
#     # walks_neis = []  # walks = {list: n} [[0, 44, 397, 266], [0, 74, 66, 196], [0, 160, 137, 358]]
#     # # walks_2_neis = []
#     # path_distance_neis = []
#     paths_root = "../path_data/"
#     if name in ['ICEWS14', 'ICEWS18', 'ICEWS05-15', "WIKI", "YAGO"]:
#         path_file = paths_root + name + "/{}_{}_{}_merw.txt".format(
#             name, num_of_walks, walk_length)
#         # path_file_neis = paths_root + name + "/{}_{}_{}_merw.txt".format(
#         #     name, num_walks_neis, walk_len_neis)
#         try:
#             with open(path_file, "r") as p:
#                 for line in tqdm.tqdm(p):
#                     info = list(map(float, line[1:-2].split(",")))
#                     path = list(map(int, info[:walk_length]))
#                     distance = list(map(int, info[walk_length:2*walk_length]))
#                     # weight = list(map(int, info[walk_length:2*walk_length]))
#                     walks.append(path)  # 获取path
#                     # walks_2.append(info[1:3])
#                     path_distance.append(distance)  # 获取这条path对应的distance距离
#                     path_weight.append(info[2*walk_length:])
#         except FileNotFoundError as fnf_error:
#             print(
#                 fnf_error, 'the file change the paths_root to where you put the sampled paths')
#         print("Opening file of paths: " + path_file)
#         print("The number of walks:", len(walks), " The number of distance:", len(path_distance))
#         # try:
#         #     with open(path_file_neis, "r") as p:
#         #         for line in tqdm.tqdm(p):
#         #             info = list(map(float, line[1:-2].split(",")))
#         #             walks_neis.append(info[:walk_len_neis])  # 获取path
#         #             path_distance_neis.append(info[walk_len_neis:])  # 获取这条path对应的distance距离
#         # except FileNotFoundError as fnf_error:
#         #     print(
#         #         fnf_error, 'the file change the paths_root to where you put the sampled paths')
#         # print("Opening file of paths: " + path_file_neis)
#         # print("The number of walks:", len(walks_neis), " The number of distance:", len(path_distance_neis))
#
#     numpy_y = np.load(paths_root + name + '/y.npy')
#     node_num = torch.from_numpy(numpy_y).to(torch.long)
#
#     # neis_path_2 = torch.tensor(walks_2, dtype=torch.long).view(
#     #     node_num, -1).to(args.cuda)
#     neis_path_all = torch.tensor(walks, dtype=torch.long).view(
#         node_num, -1).to(args.cuda)
#     # TODO args.node_num, args.num_w, args.walk_len要在args中添加进去
#     path_distance = torch.tensor(path_distance, dtype=torch.long).view(
#         node_num, num_of_walks, walk_length).to(args.cuda)
#     path_weight = torch.tensor(path_weight).view(
#         node_num, -1).to(args.cuda)
#     # neis_path_all_neis = torch.tensor(walks_neis, dtype=torch.long).view(
#     #     node_num, -1).to(args.cuda)
#     # # TODO args.node_num, args.num_w, args.walk_len要在args中添加进去
#     # path_distance_neis = torch.tensor(path_distance_neis, dtype=torch.long).view(
#     #     node_num, num_walks_neis, walk_len_neis).to(args.cuda)
#
#
#     return neis_path_all, path_distance, path_weight

def load_merw(args):
    name = args.dataset
    num_of_walks = args.num_walks
    walk_length = args.walk_len
    walks = []  # walks = {list: n} [[0, 44, 397, 266], [0, 74, 66, 196], [0, 160, 137, 358]]
    timestamps = []
    path_weight = []
    paths_root = "../path_data/"
    if name in ['ICEWS14', 'ICEWS18', 'ICEWS05-15', "GDELT"]:
        path_file = paths_root + name + "/{}_{}_{}_merw.txt".format(
            name, num_of_walks, walk_length)
        try:
            with open(path_file, "r") as p:
                for line in tqdm.tqdm(p):
                    info = list(map(float, line[1:-2].split(",")))
                    path = list(map(int, info[:walk_length]))
                    timestamp = list(map(int, info[walk_length : 2*walk_length]))
                    # walks.append(path)  # 获取path
                    # path_weight.append(info[walk_length:])
                    # walks.append(path[::-1])  # 获取path
                    # timestamps.append(timestamp[::-1])
                    # path_weight.append(info[2*walk_length:][::-1])
                    walks.append(path)  # 获取path
                    timestamps.append(timestamp)
                    path_weight.append(info[2*walk_length:])
        except FileNotFoundError as fnf_error:
            print(
                fnf_error, 'the file change the paths_root to where you put the sampled paths')
        print("Opening file of paths: " + path_file)
        print("The number of walks:", len(walks), " The number of path_weight:", len(path_weight))

    numpy_y = np.load(paths_root + name + '/y.npy')
    node_num = torch.from_numpy(numpy_y).to(torch.long)

    neis_path_all = torch.tensor(walks, dtype=torch.long).view(
        node_num, -1).to(args.cuda)
    neis_timestamps = torch.tensor(timestamps, dtype=torch.long).view(
        node_num, -1).to(args.cuda)
    path_weight = torch.tensor(path_weight).view(
        node_num, -1).to(args.cuda)
    return neis_path_all, neis_timestamps, path_weight

if __name__ == '__main__':
    save_path = "results/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.model, args.rank, args.learning_rate,args.n_hidden,args.num_walks,args.walk_len, int(time.time()))
    print("rank:",args.rank," n_hidden:", args.n_hidden, " num_walks:",args.num_walks, " walk_len:",args.walk_len, "learning_rate",args.learning_rate)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset = Dataset(args.dataset, is_cuda=True if args.gpu == 1 else False)
    fw = codecs.open("{}/log.txt".format(save_path), 'w')
    sizes = dataset.get_shape()
    model = Supercomplex(args, sizes, args.rank, is_cuda=True if args.gpu == 1 else False)
    # in case a user want to train on a non-cuda machine
    if args.gpu == 0:
        model = model.to('cpu')
    else:
        model = model.cuda()

    best_hits1 = 0
    best_res_test = {}
    opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    # TODO obtain preprocessing data
    # neis_path_all, path_distance, neis_path_all_neis, path_distance_neis = load_merw(args)
    neis_path_all, neis_timestamps, path_weight = load_merw(args)

    for epoch in range(args.max_epochs):
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        ) #得到正向与反向训练集

        model.train()
        optimizer = Train(model, args.dataset, sizes[1] // 2, opt, batch_size=args.batch_size)
        mode = "Training"
        optimizer.epoch(examples, args, mode, neis_path_all, neis_timestamps, path_weight, epoch)

        def avg_both(mr, mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            m = (mrrs['lhs'] + mrrs['rhs']) / 2.
            h = (hits['lhs'] + hits['rhs']) / 2.
            mr = (mr['lhs'] + mr['rhs']) / 2.
            return {'MR':mr, 'MRR': m, 'hits@[1,3,10]': h}
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
            model.eval()
            valid, test = [
                avg_both(*dataset.eval(model, split, args, split, neis_path_all, neis_timestamps, path_weight, epoch))
                for split in ['valid', 'test']
            ]
            print("valid: ", epoch, valid['MR'], valid['MRR'], valid['hits@[1,3,10]'])
            print("test: ", epoch, test['MR'], test['MRR'], test['hits@[1,3,10]'])
            fw.write("valid: epoch:{}, MR:{}, MRR:{}, Hist:{}\n".format(epoch, valid['MR'], valid['MRR'], valid['hits@[1,3,10]']))
            fw.write("test: epoch:{}, MR:{}, MRR:{}, Hist:{}\n".format(epoch, test['MR'], test['MRR'], test['hits@[1,3,10]']))
            if valid['hits@[1,3,10]'][0] > best_hits1:
                torch.save({'MRR':test['MRR'], 'Hist':test['hits@[1,3,10]'], 'MR':test['MR'], 'param':model.state_dict()}, '{}/best.pth'.format(save_path, args.model, args.dataset))
                print('best')
                best_hits1 = valid['hits@[1,3,10]'][0]
                best_res_test = [test['MR'], test['MRR'], test['hits@[1,3,10]']]

    fw.write("{}\t{}\t{}\t{}\t{}\n".format(best_res_test[0], best_res_test[1], best_res_test[2][0], best_res_test[2][1], best_res_test[2][2]))