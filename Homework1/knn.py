# !usr/bin/python
# -*- coding:utf-8 -*-

from numpy import mat, linalg
from operator import itemgetter
import math

'''
5.KNN
'''
def knn():
    '''
    读取向量表示文件，一行一个文本
    测试集和训练集的每个文档进行比较（需要类别和文档名），相似度最大的只比较类别
    所以把类别和文档名作为key,文档具体内容（train_word_tfidf）作为value
    :return:
    '''
    train_path = 'docVector/word_TFIDF_train_vector.txt'
    test_path = 'docVector/word_TFIDF_test_vector.txt'
    knn_result_compare_path = 'docVector/knn_result_compare.txt'

    train_doc_word_tfidf= {}  # 字典<key, value> key=cate_doc, value={{word1,tfidf1}, {word2, tfidf2},...}
    for line in open(train_path).readlines():
        train_line = line.strip('\n').split(' ')  # 每一行进行读取，去掉换行，用空格分开，这样多了一个空格
        train_word_tfidf = {}  # 字典<word, tfidf>   key:word value:tfidf值
        length = len(train_line) - 1  # 减去空格
        for i in range(2, length, 2):  # 在每个文档向量中提取(word, tfidf)存入字典 range(start,stop,step)
            train_word_tfidf[train_line[i]] = train_line[i + 1]  # key:word value:tfidf值
        train_doc_word_tfidf_key = train_line[0] + '_' + train_line[1]  # 每个文档的类别和文档名拼接，其实在tf_idf.py就可以直接以这种方式保存
        train_doc_word_tfidf[train_doc_word_tfidf_key] = train_word_tfidf  # key:cate_doc value:trainWordMap这个词典

    test_doc_word_tfidf = {}
    for line in open(test_path).readlines():
        test_line = line.strip('\n').split(' ')
        test_word_tfidf = {}
        m = len(test_line) - 1
        for i in range(2, m, 2):
            test_word_tfidf[test_line[i]] = test_line[i + 1]
        test_doc_word_tfidf_key = test_line[0] + '_' + test_line[1]
        test_doc_word_tfidf[test_doc_word_tfidf_key] = test_word_tfidf  # <类_文件名，<word, TFIDF>>

    # 遍历每一个测试文档计算与所有训练样本的距离，做分类
    count = 0
    right_count = 0
    knn_result_compare = open(knn_result_compare_path, 'w',encoding='utf-8')
    for item in test_doc_word_tfidf.items():
        train_category = compute_cate(item[1], train_doc_word_tfidf)  # 调用compute_cate做分类
        count = count + 1
        print('第 %d 个文档' % count)
        test_category = item[0].split('_')[0]  # 只取类别就可以，比较两个文档的类别
        knn_result_compare.write('%s %s\n' % (test_category, train_category))
        if test_category == train_category:
            right_count += 1
        print('%s %s right_count:%d' % (test_category, train_category, right_count))

    accuracy = float(right_count) / float(count)
    print('right_count : %d , count : %d , accuracy : %.6f' % (right_count, count, accuracy))
    return accuracy


def compute_cate(test_word_tfidf, train_doc_word_tfidf):
    '''
    :param test_word_tfidf: 测试集：{<word1,tfidf1>,<word2,tfidf2>}
    :param train_doc_word_tfidf:  训练集 {类别_文档名，<word1,tfidf1>,<word2,tfidf2>}
    :return:sorted_category_diatance[0][0] 返回与测试文档向量距离和最大的类
    '''
    cateDoc_distance = {}  # <类别_文件名,距离>
    for item in train_doc_word_tfidf.items():
        similarity = compute_similarity(test_word_tfidf, item[1])  # 调用computeSim()返回相似度
        cateDoc_distance[item[0]] = similarity  # 每一个类别_文档 与和他的距离
    sorted_cateDoc_distance = sorted(cateDoc_distance.items(), key=itemgetter(1), reverse=True)  # <类目_文件名,距离> 按照value降序排序

    # 选择相似度最大的前1/5/10/15/20/25的文档
    k = 20
    category_diatance = {}  # <类，距离和>
    for i in range(k):
        category = sorted_cateDoc_distance[i][0].split('_')[0]  # 每行第一列只取类名
        category_diatance[category] = category_diatance.get(category, 0) + sorted_cateDoc_distance[i][1] #获得key加上第i个文档对应的距离值
    sorted_category_diatance = sorted(category_diatance.items(), key=itemgetter(1), reverse=True)  # 按距离降序排序 reverse = True 降序 ， reverse = False 升序（默认）
    # print('+++++++++',sorted_category_diatance[0][0])
    return sorted_category_diatance[0][0]


def compute_similarity(test_word_tfidf, train_word_tfidf):
    '''
    计算余弦相似度：余弦值越接近1，就表明夹角越接近0度，也就是两个向量越相似
    :param test_word_tfidf: 测试集：{<word1,tfidf1>,<word2,tfidf2>}
    :param train_word_tfidf: 训练集：{<word1,tfidf1>,<word2,tfidf2>}
    :return:
    '''
    testList = []  # 测试向量与训练向量共有的词在测试向量中的tfidf值
    trainList = []  # # 测试向量与训练向量共有的词在训练向量中的tfidf值

    for word, weight in test_word_tfidf.items():
        if word in train_word_tfidf:
            testList.append(float(weight))  # float()将字符型数据转换成数值型数据
            trainList.append(float(train_word_tfidf[word]))  # 获得train里的单词的权重

    testVect = mat(testList)  # 列表转矩阵
    trainVect = mat(trainList)
    num = float(testVect * trainVect.T)
    denom = linalg.norm(testVect) * linalg.norm(trainVect)
    # print(denom)
    # print(num / denom)
    return float(num) / (1.0 + float(denom))


if __name__ == "__main__":
    knn()
