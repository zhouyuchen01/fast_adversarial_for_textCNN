# -*- coding: utf-8 -*-
# @Project : pyProj
# @FileName: adv_train_eval.py
# @Version : 1.0
# @Author  : Zhou Y.C.
# @E-mail  : zhouyuchen-01@163.com
# @Time    : 2022/4/28 11:08
# @Software: PyCharm
# @Function: 
# @Type    : 扩展包/数据预处理/统计分析/数据可视化/测试用例
# @Updates : 2021-XX-XX 
#            2021-XX-XX 
#            2021-XX-XX 
# All rights reserved.

# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter

epsilon = torch.tensor(0.1)
alpha_1 = 8 / 255
alpha_2 = 8 / 255
FREE_OPT_NUM = 4
PGD_OPT_NUM  = 4

# def clamp(X, lower_limit, upper_limit):
#     return torch.max(torch.min(X, upper_limit), lower_limit)


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    delta = torch.zeros(config.batch_size, 32, config.embed)
    delta.requires_grad = True

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            # print(device)
            # print(trains)
            # print(trains.size())
            # trains = trains.to(device)
            # labels = labels.to(device)
            if config.mode == 'Free':
                # 无任何操作的优化步骤
                # outputs = model(trains, delta[:trains[0].size(0)])
                # model.zero_grad()
                # loss = F.cross_entropy(outputs, labels)
                # loss.backward()
                # optimizer.step()

                for _ in range(FREE_OPT_NUM):
                    outputs = model(trains, delta[:trains[0].size(0)].to(device))
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    grad = delta.grad.detach()
                    delta.data = delta + epsilon * torch.sign(grad)
                    delta.data[:trains[0].size(0)] = torch.clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                    delta.grad.zero_()
            elif config.mode == 'PGD':
                # 附加PGD操作的优化步骤
                for _ in range(PGD_OPT_NUM):
                    outputs = model(trains, delta[:trains[0].size(0)])
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    grad = delta.grad.detach()
                    delta.data.uniform_(-epsilon, epsilon)
                    delta.data = delta + alpha_1 * torch.sign(grad)
                    delta.data[:trains[0].size(0)] = torch.clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                    delta.grad.zero_()
            elif config.mode == "FGSM":
                outputs = model(trains, delta[:trains[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                grad = delta.grad.detach()
                delta.data.uniform_(-epsilon, epsilon)
                delta.data = delta + alpha_1 * torch.sign(grad)
                delta.data[:trains[0].size(0)] = torch.clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                delta.grad.zero_()

                outputs = model(trains, delta[:trains[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                grad = delta.grad.detach()
                delta.data.uniform_(-epsilon, epsilon)
                delta.data = delta + alpha_1 * torch.sign(grad)
                delta.data[:trains[0].size(0)] = torch.clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                delta.grad.zero_()
            else:
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    print("Train time:", get_time_dif(start_time))
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Test time:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        plot_confusion_matrix(confusion, config.class_list, 'confusion_matrix.png', title='confusion matrix')
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap='summer')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
