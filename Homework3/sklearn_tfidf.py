# coding=utf-8

import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import codecs
from sklearn.externals import joblib

def process(origin_path, extract_text_path,extract_cluster_path):
    file = open(origin_path, 'r')  # r/w 模式，可以指定编码，也可以不指定，windows下默认是gbk编码。
    text_file = open(extract_text_path, 'w', encoding='utf-8')  # rb/wb 模式直接读取二进制，与编码没有关系，加上就报错。
    cluster_file = open(extract_cluster_path, 'w', encoding='utf-8')
    cluster = set()  # 集合可变，不重复
    text = []
    for line in file:
        line = json.loads(line)
        # print (type(decodes))
        # print (line)    #输出key前会带一个u,类似[{u'a': u'A', u'c': 3.0, u'b': [2, 4]}]
        line_clu = line['cluster']
        cluster.add(line_clu)
        cluster_file.write(str(line_clu))
        cluster_file.write('\n')

        line_text = line['text']
        text_file.write(line_text)
        text_file.write('\n')
    #     text.append(line_text)
    # print(len(text))     # 2472个文本
    # print(len(cluster))  # 总共有多少各类 89
    file.close()
    text_file.close()

def tf_idf(extract_text_path):
    '''
    回顾知识：
    TF（Term Frequency）表示某个关键词在整篇文章中出现的频率。
   IDF（InversDocument Frequency）表示计算倒文本频率。
    文本频率是指某个关键词在整个语料所有文章中出现的次数。
    倒文档频率又称为逆文档频率，它是文档频率的倒数，主要用于降低所有文档中一些常见却对文档影响不大的词语的作用。

    Scikit-Learn中TF-IDF权重计算：CountVectorizer和TfidfTransformer（见文档sklearn知识点笔记.txt）
    :param extract_text_path:
    :return:
    '''
    corpus = []
    fr = open(extract_text_path, 'r',encoding='utf-8')
    for line in fr.readlines():
        corpus.append(line.strip())
    # print(corpus)  # ['brain fluid buildup delay giffords rehab', 'trailer talk week movie rite mechanic week opportunity', 'rnc ap

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf
    # 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names() # 2472个文本中的所有词（去重之后的）：5097个

    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在第i个文本中的tf-idf权重
    weight = tfidf.toarray()   # 权重矩阵[[0. 0. 0. ... 0. 0. 0.]\n[0. 0. 0. ... 0. 0. 0.] ... 2472个文本，2472行

    joblib.dump(weight,'result/weight.pkl')  # 保存tf-idf矩阵，以便后面使用

    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))  # 特征词个数：5097（维度）
    tfidf_path= "result/tfidf_result.txt"
    result = codecs.open(tfidf_path, 'w', 'utf-8')
    for j in range(len(word)):
        result.write(word[j] + ' ')  # 保存特征词
    result.write('\n')

    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    for i in range(len(weight)):
        # print (u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            # print (word[j],weight[i][j])
            result.write(str(weight[i][j]) + ' ')
        result.write('\n')
    result.close()
    fr.close()

if __name__ == "__main__":
    process('data/Tweets.txt', 'data/Tweets_text.txt','data/Tweets_cluster.txt')
    tf_idf('data/Tweets_text.txt')