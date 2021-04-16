# coding: utf-8

from __future__ import print_function

import codecs

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import os

import numpy as np

from model import TextRNN, TextCNN
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab, read_file

base_dir = 'data'
# train_dir = os.path.join(base_dir, 'train_wordcut.txt')
# test_dir = os.path.join(base_dir, 'test_wordcut.txt')
# val_dir = os.path.join(base_dir, 'val_wordcut.txt')
train_dir = os.path.join(base_dir, 'little_train_wordcut.txt')
test_dir = os.path.join(base_dir, 'little_test_wordcut.txt')
val_dir = os.path.join(base_dir, 'little_val_wordcut.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')
cat_dir = os.path.join(base_dir, 'little_categories.txt')


# 挑选小数据量进行本地测试
# def save_little_data():
#     max_size = 10000
#     labels, contents = read_file('data/little_train_wordcut.txt')
#     f = codecs.open('data/little_train_wordcut.txt', 'w', 'utf-8')
#     f.write('\n'.join(train_wordcut_list))
#     f.close()
#
#     # 将分词后的训练集、评估集和测试集保存在本地
#
#     f = codecs.open('data/little_val_wordcut.txt', 'w', 'utf-8')
#     f.write('\n'.join(val_wordcut_list))
#     f.close()
#     f = codecs.open('data/little_test_wordcut.txt', 'w', 'utf-8')
#     f.write('\n'.join(test_wordcut_list))
#     f.close()
#     # 生成作者名到id的字典并保存到本地
#     cat_to_id = dict(zip(categories, range(len(categories))))
#     f = codecs.open('data/little_categories.txt', 'w', 'utf-8')
#     for k, v in cat_to_id.items():
#         f.write(k + '\t' + str(v) + '\n')
#     f.close()
#     # 生成词向量
#     vocab_dir = 'data/little_vocab.txt'
#     save_vocab(vocab_dir)


def train():
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, 600)  # 获取训练数据每个字的id和对应标签的oe-hot形式
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, 600)
    # 使用LSTM或者CNN
    # model = TextRNN(len(word_to_id), len(cat_to_id))
    model = TextCNN(len(word_to_id), len(cat_to_id))
    # 选择损失函数
    # Loss = nn.MultiLabelSoftMarginLoss()
    # Loss = nn.BCELoss()
    Loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0
    num_epochs = int(len(y_train) / 100) + 1
    for epoch in range(num_epochs):
        batch_train = batch_iter(x_train, y_train, 100)
        for x_batch, y_batch in batch_train:
            x = x_batch
            y = y_batch
            x = torch.LongTensor(x)
            y = torch.Tensor(y)
            # y = torch.LongTensor(y)
            x = Variable(x)
            y = Variable(y)
            out = model(x)
            loss = Loss(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).numpy())
        # 对模型进行验证
        if (epoch + 1) % 20 == 0:
            batch_val = batch_iter(x_val, y_val, 100)
            for x_batch, y_batch in batch_val:
                x = np.array(x_batch)
                y = np.array(y_batch)
                x = torch.LongTensor(x)
                y = torch.Tensor(y)
                # y = torch.LongTensor(y)
                x = Variable(x)
                y = Variable(y)
                out = model(x)
                loss = Loss(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).numpy())
                if accracy > best_val_acc:
                    torch.save(model.state_dict(), 'model_params.pkl')
                    best_val_acc = accracy
                print(accracy)


if __name__ == '__main__':
    # 获取文本的类别及其对应id的字典
    categories, cat_to_id = read_category(cat_dir)
    # 获取训练文本中所有出现过的字及其所对应的id
    words, word_to_id = read_vocab(vocab_dir)
    # 获取字数
    vocab_size = len(words)
    train()
