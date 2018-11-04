# !usr/bin/python
# -*- coding:utf-8 -*-


import os
import math


'''
4.计算tf-idf,向量表示
'''

def compute_idf():
    '''
    计算所有单词的IDF值 ：文本总数，出现某单词的文本总数（某单词出现新哪个文档里）
    训练集和测试集分开统计，注意修改原路径、写入路径和公式中的文档总数
    :return:
    '''
    src_file = 'featureWord/testFilter'
    # src_file = 'featureWord/trainFilter'

    word_doc_dict = {}  # <word, set(docM,...,docN)> 存放单词，以及单词出现的文档
    idf_perWord_dict = {}  # <word, IDF值>
    class_lists = os.listdir(src_file)
    for class_list in class_lists:
        class_list_path = src_file + '/' + class_list
        text_files = os.listdir(class_list_path)  # 每一个具体文档
        for text_file in text_files:
            text_file_path = class_list_path + '/' + text_file  # 文档路径
            for line in open(text_file_path).readlines():
                word = line.strip('\n')
                if word in word_doc_dict.keys():
                    word_doc_dict[word].add(text_file)  # 此单词出现在此文档就把此文档写入
                else:
                    word_doc_dict.setdefault(word, set())  # dict.setdefault(key, default=None) set集合set：保存单词word出现过的文档
                    word_doc_dict[word].add(text_file)
        print('just finished' + class_list)
    for word in word_doc_dict.keys():
        count_doc = len(word_doc_dict[word])  # 某个词出现在多少个文档中 set中的文档个数

        idf = math.log((3772)/(count_doc))  #测试  (文档统数量在splitData.py里有统计)
        # idf = math.log((15056) / (count_doc))  # 训练  (文档统数量在splitData.py里有统计)

        idf_perWord_dict[word] = idf

    fw = open('IDF_perWord/idf_test.txt', 'w')
    # fw = open('IDF_perWord/idf_train.txt', 'w')

    for word, idf in idf_perWord_dict.items():
        fw.write('%s %.6f\n' % (word, idf))
    fw.close()


def compute_tf_tfidf():
    '''
    读取每个的idf
    计算tf，同时计算tfidf
    生成训练集和测试集的文档向量，向量形式<category, doc, (word1, tfidf1), (word2, tfidf2),...> 存入文件
    :return:
    '''
    idf_perWord_dict = {}  # <word, IDF值> 从文件中读入后的数据保存在字典中

    for line in open('IDF_perWord/idf_test.txt').readlines():
    # for line in open('IDF_perWord/idf_train.txt').readlines():
        (word, idf) = line.strip('\n').split(' ')
        idf_perWord_dict[word] = idf

    src_path = 'featureWord/testFilter'
    # src_path = 'featureWord/trainFilter'

    test_vertor_path = "docVector/" + 'word_TFIDF_test_vector.txt'
    # train_vertor_path = "docVector/" + 'word_TFIDF_train_vector.txt'

    test_vertor = open(test_vertor_path, 'w', encoding='utf-8')
    # train_vertor = open(train_vertor_path, 'w', encoding='utf-8')

    class_lists = os.listdir(src_path)
    for class_list in class_lists:
        class_path = src_path + '/' + class_list
        text_lists = os.listdir(class_path)
        for text_list in text_lists:
            tf_perWord_dict = {}  # <word, 文档里该word出现的次数>
            count_words = 0  # 统计每个文档里的单词总数
            text_path = class_path + '/' + text_list
            for line in open(text_path).readlines():
                count_words = count_words + 1  # 统计每个文档里总单词数
                word = line.strip('\n')
                tf_perWord_dict[word] = tf_perWord_dict.get(word, 0) + 1  #每个单词

            test_vertor.write('%s %s ' % (class_list, text_list))  # 写入类别，文档
            # train_vertor.write('%s %s ' % (class_list, text_list))  # 写入类别，文档

            for word, count in tf_perWord_dict.items():
                tf = float(count) / float(count_words)
                tf_idf = tf * float(idf_perWord_dict[word])

                test_vertor.write('%s %f ' % (word, tf_idf))  # 继续写入类别cate下文档doc下的所有单词及它的TF-IDF值
                # train_vertor.write('%s %f ' % (word, tf_idf))  # 继续写入类别cate下文档doc下的所有单词及它的TF-IDF值

            test_vertor.write('\n')
            # train_vertor.write('\n')

    test_vertor.close()
    # train_vertor.close()

if __name__ == "__main__":
    # compute_idf()
    compute_tf_tfidf()
