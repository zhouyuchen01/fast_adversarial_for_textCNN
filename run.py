# -*- coding: utf-8 -*-
# @Project : pyProj
# @FileName: run3.py
# @Version : 1.0
# @Author  : Zhou Y.C.
# @E-mail  : zhouyuchen-01@163.com
# @Time    : 2022/4/28 11:06
# @Software: PyCharm
# @Function:
# @Type    : 扩展包/数据预处理/统计分析/数据可视化/测试用例
# @Updates : 2021-XX-XX
#            2021-XX-XX
#            2021-XX-XX
# All rights reserved.

# coding: UTF-8
import time
import torch
import numpy as np
from adv_train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='TextCNN', type=str, help='Only TextCNN is avaiable')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--mode', default='Free', type=str, required=True, help='choose mode: BaseLine, Free, PGD, FGSM')
parser.add_argument('--seed', default=1, type=int, help='set the seed')

args = parser.parse_args()


if __name__ == '__main__':

    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif

        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding, args.mode)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)

    train(config, model, train_iter, dev_iter, test_iter)