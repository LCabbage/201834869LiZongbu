# !usr/bin/python
# -*- coding:utf-8 -*-

import os
import random
import shutil

'''
2、划分数据集
随机选取80%作为训练集，20%测试集，分别保存到split/splitTrain/和split/splitTest/文件夹下
并在splitDoc文件夹下在创建splitTrain.txt splitTest.txt文档，用来保存文档编号和对应的类别
切记：运行保存完不要再运行，后面操作都是基于此随机文档
'''
path = 'preprocess/'
def split_data():
    # fr = open(classifyRightCate, 'w')
    train_path = 'split/split_train/'
    test_path = 'split/split_test/'

    fw_train = open('splitDoc/splitTrain.txt', 'w', encoding='utf-8')  # 存编号和所属类别
    fw_test = open('splitDoc/splitTest.txt', 'w', encoding='utf-8')

    i = 0  # 统计训练总文档多少个
    j = 0  # 统计测试总文档多少个
    class_list = os.listdir(path)  # 每个类文件名 #alt.atheism
    for doc_class in class_list:
        class_path = os.path.join(path, doc_class)  # 拼接路径 /alt.atheism
        file_lists = os.listdir(class_path)  # 每一个文档 49960 51060...
        num = len(file_lists)  # 每类里的文档数目
        # print(num)
        train_num = int(num * 0.8)
        train_class_list = random.sample(file_lists, train_num)  # 随机选取的每一个文档 从每类的类文件里，随机选择80%作为训练数据
        # print(train_class_list)
        for file_list in file_lists:
            src_path = class_path + '/' + file_list
            # print(ori_path)
            if file_list in train_class_list:
                train_list_path = train_path + doc_class
                fw_train.write('%s %s\n' % (file_list, doc_class))
                i = i + 1
                # print(train_list_path)  # train/alt.atheism/51181
                if os.path.exists(train_list_path) == False:
                    os.makedirs(train_list_path)
                else:
                    print('%s exists' % train_list_path)
                shutil.copy(src_path, train_list_path + '/' + file_list)
            else:
                test_list_path = test_path + doc_class
                # print(test_list_path)
                fw_test.write('%s %s\n' % (file_list, doc_class))
                j = j + 1
                if os.path.exists(test_list_path) == False:
                    os.makedirs(test_list_path)
                else:
                    print('%s exists' % test_list_path)
                shutil.copy(src_path, test_list_path + '/' + file_list)
    print("训练总文档数：" , i)   # 15056
    print("测试总文档数：" , j)   # 3772
    fw_train.close()
    fw_test.close()


if __name__ == "__main__":
    split_data()
