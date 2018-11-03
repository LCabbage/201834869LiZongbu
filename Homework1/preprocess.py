# !/usr/bin/python
# -*- coding:utf-8 -*-

import os
import random
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

'''
1、预处理，并将处理过的数据保存到‘preprocess/’下
'''

# 分词 Tokenization
def tokenization(contenFile_path):
    '''
    预处理：tokenization,stemming/normalization,stopwords,a-zA-Z外的用空格替换
    :param contenFile_path:
    :return:
    '''
    with open(contenFile_path, 'rb') as fr:
        file_context = fr.read().decode('utf-8', 'ignore')  # 读取整个文件

        # 文件编码问题
        # encode_type = chardet.detect(file_context)
        # file_context = file_context.decode(encode_type['encoding']) #进行相应解码  从str到bytes:调用方法encode(),从bytes到str:调用方法decode()
        # Normalization
        #
    text = file_context.lower() #全部化为小写
    text = re.sub(r"[^a-zA-Z]"," ",text)  #a-zA-Z外的用空格替换
    #print(text)
    #words = text.split()   #拆分单词
    words = word_tokenize(text)
    #stemmer = PorterStemmer()
    words = [PorterStemmer().stem(w) for w in words] #词干提取
    # print(words)
    # lem = [WordNetLemmatizer().lemmatize(w) for w in words]
    # print(lem)
    stopWords = stopwords.words("english") # 英文停用词
    wordDict = []
    for w in words:
        if w not in stopWords:
            wordDict.append(w)
    #print(wordDict)
    return wordDict



def preprocessData(path):
    # word_dict = {}
    # new_word_dict = {}
    class_list = os.listdir(path)  # 每个类文件名 #alt.atheism
    for file in class_list:
        class_path = os.path.join(path, file)  # Data/20news-18828/alt.atheism
        file_list = os.listdir(class_path)  # 每一个文本文件
        #print(file_list)   #['49960', '51060', '51119', '51120', '51121', '51122', '51123'..]
        preprocess_file = 'preprocess/' + file  #处理之后保存路径
        if os.path.exists(preprocess_file) == False:
            os.makedirs(preprocess_file)
        else:
            print('%s exists' % preprocess_file)
        for text_file in file_list:  #遍历文本文件
            #print(text_file)
            text_file_path = os.path.join(class_path+'/', text_file)  #文本文件路径
            # print(text_file_path)
            preprocess_content = tokenization(text_file_path)  # 预处理
            preprocess_text_file_path= preprocess_file +'/'+ text_file # 对预处理进行保存
            fw = open(preprocess_text_file_path, 'w',encoding='utf-8')
            for word in preprocess_content:
                fw.write('%s\n' % word)   #每个单词一行便于后面构建词典等操作
                #word_dict[word] = word_dict.get(word, 0.0) + 1.0  #统计词频放在字典里
            #print("===============",word_dict)
            fw.close()

# def read_content(read_path):
#     '''
#     Read text content from a file
#     :param readfile_path:
#     :return:
#     '''
#     with open(read_path, 'rb') as fr:
#         content = fr.read().decode('utf-8', 'ignore')
#         #print(content)
#     return content

if __name__ == "__main__":
    preprocessData('Data/20news-18828/')
    #data = tokenization('Data/20news-18828/alt.atheism/49960')
    # print(data)
