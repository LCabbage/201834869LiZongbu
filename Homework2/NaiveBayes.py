# -*- coding:utf-8 -*-

import os
import math

def word_count(path):
    '''
    需要统计 每类中每个单词出现的次数，每类中所有单词总数
    :return:
    '''
    cate_each_word_num = {}  # 类别_单词，每个单词出现次数
    cate_total_words = {}  # 类别，每类总单词数
    cate_lists = os.listdir(path)
    # print(cate_lists)   # 'alt.atheism', 'comp.graphics', '
    for cate_list in cate_lists:  # 每一类

        count = 0  # 统计每个类里的总单词数，每个文档一行加一次
        file_lists_path = path + '/' + cate_list
        file_lists = os.listdir(file_lists_path)
        # print(file_lists)
        for file_list in file_lists:
            file_path = file_lists_path + '/' + file_list  # 具体文档路径
            with open(file_path, 'r') as fr:
                for line in fr.readlines():
                    count += 1
                    word = line.strip('\n')
                    key = cate_list + '_' + word
                    cate_each_word_num[key] = cate_each_word_num.get(key, 0) + 1  # 每类里每个单词出现次数
        cate_total_words[cate_list] = count  # 每一类总单词数（重复）
        print(cate_list, '单词总数为：', count)
    print(len(cate_each_word_num))
    return cate_each_word_num, cate_total_words

def NB_process(train_path, test_path, classify_result):
    '''
    分类：
    :param train_path:
    :param test_path:
    :param classify_result: 分类结果<文档 所属类别 预测类别>
    :return:
    '''
    fw = open(classify_result, 'w')
    # 返回类k下词C的出现次数，类k总词数
    cate_each_word_num, cate_total_words = word_count(train_path)

    # 被统计的词表的词语数量：训练集的总词数(分母1)
    trainTotalNum = sum(cate_total_words.values())  # 所有类的总和，训练集总单词数
    print('trainTotalNum: %d' % trainTotalNum)

    # 对测试集分类
    test_cate_lists = os.listdir(test_path)
    for test_cate_list in test_cate_lists:
        test_cate_path = test_path + '/' + test_cate_list
        test_file_lists = os.listdir(test_cate_path)
        for test_file_list in test_file_lists:
            # 测试集的每个文档分别与训练集的所有类比较，选择概率最大的那个
            testFilesWords = []
            test_file_path = test_cate_path + '/' + test_file_list
            lines = open(test_file_path).readlines()
            for line in lines:
                word = line.strip('\n')
                testFilesWords.append(word)  # 训练集每个文档的的所有单词

            max_p = 0.0
            train_cate_lists = os.listdir(train_path)
            for train_cate_list in train_cate_lists:
                p = compute_prob(train_cate_list, testFilesWords, cate_total_words, cate_each_word_num, trainTotalNum)

                if train_cate_list == train_cate_lists[0]:
                    max_p = p
                    best_cate = train_cate_list
                    continue
                if p > max_p:
                    max_p = p
                    best_cate = train_cate_list
            fw.write('%s %s %s\n' % (test_file_list, test_cate_list, best_cate))
    fw.close()


def compute_prob(train_cate_list, testFilesWords, cate_total_words, cate_each_word_num, trainTotalNum):
    '''
    计算条件概率和先验概率，使用的是多项式模型，并且使用了平滑技术（避免0的情况）和取对数（加快计算速度）
    条件概率 =（某个类里的某个单词出现的次数+1.0）/（某个类单词总数+训练样本中所有类单词总数）
    先验概率 =（某个类单词总数）/（训练样本中所有类单词总数）
    :param train_cate_list:    训练集类别
    :param testFilesWords:     测试集某文档单词
    :param cate_total_words:   某类别所有单词  <类别，类别单词总数>
    :param cate_each_word_num: 某类别里某个单词出现的次数  <类别_单词，单词的次数>
    :param trainTotalNum:      被统计的词表的词语数量：训练集的总词数(分母1)
    :return:
    '''
    Log_total_word = 0

    # 某类下的单词总数 （分母2）
    wordNumInCate = cate_total_words[train_cate_list]  # <类别，单词总数>  (包括重复)

    # 每个文档的每个单词在测试集的每个类别中出现的次数总和
    for i in range(len(testFilesWords)):
        keyName = train_cate_list + '_' + testFilesWords[i]  # 测试集中某个文档的某个词
        if keyName in cate_each_word_num:
            testFileWordNumInCate = cate_each_word_num[keyName]  # 训练集中某类下某个单词出现的次数
        else:
            testFileWordNumInCate = 0.0
        p_conditional = (testFileWordNumInCate + 1.0) / (wordNumInCate + trainTotalNum)  # 计算在某个文档中中出现某个词的概率
        Log_conditional = math.log(p_conditional)  # 取对数
        Log_total_word = Log_total_word + Log_conditional  # 文档所有词求和

    # 先验概率
    Log_prior = math.log(wordNumInCate / trainTotalNum)

    p = Log_total_word + Log_prior
    return p


if __name__ == "__main__":
    # word_count('Data/featureWord/trainFilter')
    NB_process('Data/split/split_train', 'Data/split/split_test', 'classify_result')
    # getCateWordsProb('Data/featureWord/trainFilter')
