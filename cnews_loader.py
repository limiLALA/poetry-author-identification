# coding: utf-8

import sys
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    labels, contents = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if len(content) > 0:
                    labels.append(native_content(label))
                    contents.append(native_content(content).strip().split(' '))
            except:
                pass
    return labels, contents


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    _, data_train = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # # words = open_file(vocab_dir).read().strip().split('\n')
    # with open_file(vocab_dir) as fp:
    #     # 如果是py2 则每个值都转化为unicode
    #     words = [native_content(_.strip()) for _ in fp.readlines()]
    # word_to_id = dict(zip(words, range(len(words))))
    words, vectors = read_file(vocab_dir)
    # vectors = [[float(i) for i in v] for v in vectors]
    # word_to_id = dict(zip(words, vectors))  # 词典，每个词对应一个词向量
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category(cat_dir):
    """读取分类目录，固定"""
    cat, id = read_file(cat_dir)
    id = [int(x[0]) for x in id]  # 将str类型的id转为int
    cat_to_id = dict(zip(cat, id))  # 词典，每个词对应一个词向量
    return cat, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    labels, contents = read_file(filename)  # 读取训练数据的每一句话及其所对应的类别
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])  # 将每句话id化
        label_id.append(cat_to_id[labels[i]])  # 每句话对应的类别的id
    #
    # # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    # x_pad = np.array(data_id)
    y_pad = np.asarray(pd.get_dummies(labels))
    # y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    #
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    if not os.path.exists(path):  # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False
