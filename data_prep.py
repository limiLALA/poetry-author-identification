"""数据预处理"""
import codecs
from tqdm import tqdm_notebook
import re
import jieba
import pyodbc
import pandas as pd

# 读取accdb语料库
from cnews_loader import mkdir
# from word2vec import save_vocab

file_path = r"D:\XMU\1-4.大四\大四下\毕业设计\语料库\5in1唐宋诗词汇总.accdb"  # file_path是access文件的绝对路径。
conn = pyodbc.connect(u'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + file_path)  # 链接数据库
cursor = conn.cursor()  # 创建游标
tablename = ["全唐诗jt", "全宋诗jt"]  # tablename是access数据库中的表名
cursor.execute("select [author],[verse_zl] from [{}] order by [author]".format(tablename[0]))
data = cursor.fetchall()  # 获取数据库中表的全部数据，n行2列
cursor.execute("select [author],[verse_zlk_BDJZ] from [{}] order by [author]".format(tablename[1]))
data.extend(cursor.fetchall())  # 加入全宋诗的诗集
cursor.close()  # 关闭游标
conn.close()  # 关闭链接

# 将查找出来的数据存入txt备用
mkdir('data/content_wordcut_by_author')
# mkdir('data/verse_by_author')
last_author = data[0][0]  # 上一首诗的作者
single_author_verse_list = []  # 只存储单个作者的诗，格式[[author][verse]]
single_author_verse_wordcut_list = []  # 只存储单个作者的诗（已分词），格式[[author][verse]]
train_list = []  # 训练集
val_list = []  # 评估集
test_list = []  # 测试集
train_wordcut_list = []  # 训练集（已分词）
val_wordcut_list = []  # 评估集（已分词）
test_wordcut_list = []  # 测试集（已分词）
categories = []  # 种类，即作者名
single_author_content_wordcut_list = []  # 单个作者的诗内容分词后的结果
vocab_list = []  # 词语
for verse in data:
    if len(verse[0]) < 2:  # 不规范数据
        continue
    if len(last_author) < 2:
        last_author = verse[0]
    if len(categories) > 2:
        break
    if verse[0] != last_author:  # 上一作者的诗集全部遍历完成
        # 剔除作诗数量过少的作者
        if len(single_author_verse_list) > 2000:
            # # 按作者归类，将诗集存储在本地
            # f = codecs.open('data/verse_by_author/{}.txt'.format(last_author), 'w', 'utf-8')
            # f.write('\n'.join(single_author_verse_list))
            # f.close()
            # 分割训练和测试集
            threshold1 = int(0.8 * len(single_author_verse_list))
            threshold2 = int(0.9 * len(single_author_verse_list))
            # train_list.extend(single_author_verse_list[:threshold1])
            # val_list.extend(single_author_verse_list[threshold1:threshold2])
            # test_list.extend(single_author_verse_list[threshold2:])
            # 分词
            for verse in single_author_verse_list:
                sentences_list = []  # 始终都仅装入一首诗
                text_list = re.findall(r'[\u4e00-\u9fff]+', verse[1])  # 找到内容中所有汉字
                for text in text_list:
                    # 将单句进行分词，用一个空格分隔
                    tmp_wordcut_list = jieba.lcut(text)
                    vocab_list.extend(tmp_wordcut_list)
                    sentence = ' '.join(tmp_wordcut_list)
                    sentences_list.append(sentence)
                sentences = '   '.join(sentences_list)  # 一首诗的分词结果，用空格替代标点符号
                single_author_verse_wordcut_list.append(verse[0] + '\t' + sentences)  # 用tab键分割作者和分词后的诗句内容
                single_author_content_wordcut_list.append(sentences)  # 将分词后的诗str加入列表
            # 分割训练和测试集
            train_wordcut_list.extend(single_author_verse_wordcut_list[:threshold1])
            val_wordcut_list.extend(single_author_verse_wordcut_list[threshold1:threshold2])
            test_wordcut_list.extend(single_author_verse_wordcut_list[threshold2:])
            # 存储作者
            categories.append(last_author)
            # # 将分词后的诗内容按作者保存，用于gensim生成词向量
            # f = codecs.open('data/content_wordcut_by_author/{}.txt'.format(last_author), 'w', 'utf-8')
            # f.write('    \n'.join(single_author_content_wordcut_list))
            # f.close()
        # 清空容器
        single_author_verse_list = []  # 清空上个作者的诗
        single_author_verse_wordcut_list = []  # 清空上个作者的诗（已分词）
        single_author_content_wordcut_list = []  # 清空上个作者的诗内容（已分词）
        last_author = verse[0]  # 更新上一个作者
    verse[1] = verse[1].replace('\n', '')
    # single_author_verse_list.append('\t'.join(verse))  # 用tab符分割作者和诗句内容
    single_author_verse_list.append(verse)  # 用tab符分割作者和诗句内容


# # 将训练集、评估集和测试集存在本地
# f = codecs.open('data/train.txt', 'w', 'utf-8')
# f.write('\n'.join(train_list))
# f.close()
# f = codecs.open('data/val.txt', 'w', 'utf-8')
# f.write('\n'.join(val_list))
# f.close()
# f = codecs.open('data/test.txt', 'w', 'utf-8')
# f.write('\n'.join(test_list))
# f.close()
# 将分词后的训练集、评估集和测试集保存在本地
f = codecs.open('data/little_train_wordcut.txt', 'w', 'utf-8')
f.write('\n'.join(train_wordcut_list))
f.close()
f = codecs.open('data/little_val_wordcut.txt', 'w', 'utf-8')
f.write('\n'.join(val_wordcut_list))
f.close()
f = codecs.open('data/little_test_wordcut.txt', 'w', 'utf-8')
f.write('\n'.join(test_wordcut_list))
f.close()
# 生成作者名到id的字典并保存到本地
cat_to_id = dict(zip(categories, range(len(categories))))
f = codecs.open('data/little_categories.txt', 'w', 'utf-8')
for k, v in cat_to_id.items():
    f.write(k + '\t' + str(v) + '\n')
f.close()

# # 生成词向量
# vocab_dir = 'data/vocab.txt'
# save_vocab(vocab_dir)

vocab_list = list(set(vocab_list))
vocab_list = [vocab_list[i]+'\t'+str(i) for i in range(len(vocab_list))]
f = codecs.open('data/little_vocab.txt', 'w', 'utf-8')
f.write('\n'.join(vocab_list))
f.close()
