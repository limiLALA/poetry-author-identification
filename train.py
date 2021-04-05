# coding:utf-8
# import random
# import torch.nn as nn
# import torch.optim as optim
import dataHandler
# from model import PoetryModel
# from utils import *
# import cPickle as p
import re
import jieba
from collections import Counter

# 生成基于诗词语料的字向量和词向量
data = dataHandler.parseRawData()  # All if author=None
# print(type(data))
seg_list = []  # 切分好的字向量和词向量
sentences = ''
sentences_list = []
paragraphs_list = []
for p in data:
    text_list = re.findall(r'[\u4e00-\u9fff]+', p)  # 找到所有汉字
    for text in text_list:
        # 将单句进行分词，用一个空格分隔
        sentence = ' '.join(jieba.lcut(text, cut_all=True))
        sentences_list.append(sentence)
    sentences = '   '.join(sentences_list)  # 一首诗的预处理结果
    paragraphs_list.append(sentences)
paragraphs = '    \n'.join(paragraphs_list)
f = open('./data/all.txt','w')
f.write(paragraphs)
f.close()

# 统计各个字向量和词向量出现的次数
# word_dic = Counter(seg_list)
# word_dic = sorted(word_dic.items(), key=lambda item:item[1], reverse=True)  # 倒序
# # print("word_dic={}, len={}".format(word_dic, len(word_dic)))
# new_word_dic = {k:v for k,v in word_dic if v>10}
# print("new_word_dic={}, len={}".format(new_word_dic, len(new_word_dic)))

# CBOW模型生成词向量





# data = dataHandler.parseRawData(author="李白",constrain=5)  # All if author=None
# random.shuffle(data)
# for s in data:
    # print(s)
# word_to_ix = {}
'''
for sent in data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
word_to_ix['<EOP>'] = len(word_to_ix)
word_to_ix['<START>'] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)

print "VOCAB_SIZE:", VOCAB_SIZE
print "data_size", len(data)

for i in range(len(data)):
    data[i] = toList(data[i])
    data[i].append("<EOP>")
# save the word dic for sample method
p.dump(word_to_ix, file('wordDic', 'w'))

# save all avaible word
# wordList = open('wordList','w')
# for w in word_to_ix:
#     wordList.write(w.encode('utf-8'))
# wordList.close()

model = PoetryModel(len(word_to_ix), 256, 256);
model.cuda()  # running on GPU,if you want to run it on CPU,delete all .cuda() usage.
optimizer = optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.0001)
criterion = nn.NLLLoss()

one_hot_var_target = {}
for w in word_to_ix:
    one_hot_var_target.setdefault(w, make_one_hot_vec_target(w, word_to_ix))

epochNum = 10
TRAINSIZE = len(data)
batch = 100
def test():
    v = int(TRAINSIZE / batch)
    loss = 0
    counts = 0
    for case in range(v * batch, min((v + 1) * batch, TRAINSIZE)):
        s = data[case]
        hidden = model.initHidden()
        t, o = makeForOneCase(s, one_hot_var_target)
        output, hidden = model(t.cuda(), hidden)
        loss += criterion(output, o.cuda())
        counts += 1
    loss = loss / counts
    print "=====",loss.data[0]
print "start training"
for epoch in range(epochNum):
    for batchIndex in range(int(TRAINSIZE / batch)):
        model.zero_grad()
        loss = 0
        counts = 0
        for case in range(batchIndex * batch, min((batchIndex + 1) * batch, TRAINSIZE)):
            s = data[case]
            hidden = model.initHidden()
            t, o = makeForOneCase(s, one_hot_var_target)
            output, hidden = model(t.cuda(), hidden)
            loss += criterion(output, o.cuda())
            counts += 1
        loss = loss / counts
        loss.backward()
        print epoch, loss.data[0]
        optimizer.step()
    test()
torch.save(model, 'poetry-gen.pt')
'''