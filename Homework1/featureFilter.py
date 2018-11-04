# !/usr/bin/python
# -*- coding:utf-8 -*-

import os

'''
3、创建词典，保存到对应文档里，过滤低频词
注意：三个函数路径的一致修改
'''

test_path = 'split/split_test'
train_path = 'split/split_train'
def create_dict():
    '''
    分别读取训练和测试集文件，统计词频，去掉小于某个次数的词，对词典排序，构建词典
    将词典分别保存到文件'dictionary/'
    :return:
    '''
    word_dict = {}
    new_word_dict = {}
    # 统计词频
    # class_list = os.listdir(test_path)  # 每个类文件名 #alt.atheism
    class_list = os.listdir(train_path)

    for file in class_list:
        # class_path = os.path.join(test_path, file)
        class_path = os.path.join(train_path, file)

        file_list = os.listdir(class_path)  # 每一个文本文件
        #  print(file_list)
        for text_file in file_list:
            text_file_path = os.path.join(class_path + '/', text_file)
            with open(text_file_path, 'r') as fr:
                for line in fr.readlines():
                    word = line.strip('\n')  # s.strip(rm) 删除s字符串中开头、结尾处，位于rm删除序列的字符
                    word_dict[word] = word_dict.get(word, 0.0) + 1.0  # 统计词频放在字典里
    # print(word_dict)
    print('originalDict size : %d' % len(word_dict))  # 训练数据：79268  测试数据：40471

    # 对字典遍历，根据词频进行筛选，去掉词频小于等于4的词，对词典排序
    for key, value in word_dict.items():  # word_dict.items返回 key value值
        if value > 4:
            new_word_dict[key] = value
    sorted_dict = sorted(new_word_dict.items())  # 按key排序
    # sorted_dict = sorted(new_word_dict.items(), key=lambda e:e[1])  #value排序 默认升序
    print('newDictsize : %d' % len(sorted_dict))  # 训练数据：24202  测试数据：11155
    # print(sorted_dict)   #{key,value}
    return sorted_dict

# 把词典里的数据写到文件中
def save_dict():
    testDict_path = 'dictionary/testDict.txt'
    trainDict_path = 'dictionary/trainDict.txt'

    sorted_dict = create_dict()
    # di_path = open(testDict_path, 'w')
    di_path = open(trainDict_path, 'w')

    for item in sorted_dict:
        di_path.write('%s %d\n' % (item[0], item[1]))

def select_feature_word():
    '''
    （create_dict()只是把词频小于等于1的词保存到一个词典中，但是每个文档没有过滤保存）
    特征提取：
    对预处理得到的文档进行过滤，把不在词典里的词（低频词）过滤掉
    然后保存到对应的各个文档中
    '''
    # path = test_path
    path = train_path

    compare_dict = {}  #保存上面返回的词典
    sorted_dict1 = create_dict() #把函数返回的值给sorted_dict1
    for i in range(len(sorted_dict1)):
        compare_dict[sorted_dict1[i][0]] = sorted_dict1[i][0]

    class_list = os.listdir(path)  # 每个类文件名 #alt.atheism
    for file in class_list:
        class_path = os.path.join(path, file)  # preprocess/alt.atheism
        file_list = os.listdir(class_path)  # 每一个文本文件

        # featureWord_file = 'featureWord/testFilter/' + file  # 提取特征之后的类文件路径
        featureWord_file = 'featureWord/trainFilter/' + file

        if os.path.exists(featureWord_file) == False:
            os.makedirs(featureWord_file)
        else:
            print('%s exists' % featureWord_file)
        for text_file in file_list:  # 遍历文本文件
            text_file_path = class_path + '/'+ text_file  # 文本文件路径
            featureWord_file_path = featureWord_file + '/' + text_file  # 提取特征之后的文档路径
            fw = open(featureWord_file_path, 'w', encoding='utf-8')
            with open(text_file_path,'r') as fr:  #打开预处理的文档，读取数据，如果在dictionary.txt,保存到新的文档里
                for line in fr.readlines():
                    word = line.strip('\n')
                    if word in compare_dict.keys():
                        fw.write('%s\n' % word)
            fw.close()

if __name__ == "__main__":
    save_dict()
    select_feature_word()
