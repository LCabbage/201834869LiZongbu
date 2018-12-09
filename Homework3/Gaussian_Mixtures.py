# coding = utf-8
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import time
import numpy as np

def gaussian_mixtures():
    weight = joblib.load('result/weight.pkl')
    # print(weight)
    fw = open('result/result.txt', 'a', encoding='utf-8')
    fw.write('Gaussian mixtures')
    fw.write('\n')

    clf = GaussianMixture(n_components=89,covariance_type='diag') # 必须设置covariance_type这个参数
    time_start = time.time()
    s = clf.fit(weight)
    time_run = time.time() - time_start

    labels = clf.predict(weight)

    print(s)
    print(np.unique(labels))

    # print(clf.weights_)
    # print(clf.means_)

    # 每个样本所属的类别
    # labels = clf.labels_  # 高斯混合模型没有此属性
    print(labels)


    pred_label = []  # 存储2742个文本的预测标签
    a = {}  # key:类别标签,value:此类的所有文档
    i = 1
    while i <= len(labels):
        # print(i, clf.labels_[i - 1])  # 第几个文本，对应的类别标签
        pred_label.append(labels[i - 1])

        if labels[i - 1] not in a.keys():
            a[labels[i - 1]] = []
        else:
            a[labels[i - 1]].append(i)

        i = i + 1
    print(a)
    print('pred_lable:', pred_label)

    true_lable = []
    file = open('data/Tweets_cluster.txt', 'r')
    for line in file:
        line = line.strip('\n')
        true_lable.append(int(line))
    print('true_lable:', true_lable)

    # 性能评估：NMI（标准化互信息）
    nmi = metrics.normalized_mutual_info_score(true_lable, labels)
    print('NMI值为:%f' % ( nmi))  # 结果越相似NMI值应接近1；算法结果很差则NMI值接近0
    print('运行时间:', time_run)

    fw.write('\t')
    fw.write('NMI值为:' + str(nmi) + ' ' + '运行时间:' + str(time_run))

    localtime = time.localtime(time.time())  # time.localtime()方法，作用是格式化时间戳为本地的时间
    time_format = time.strftime('%Y-%m-%d %H:%M:%S', localtime)  # 格式化制定形式
    fw.write(' ' + '本地时间：' + time_format)
    fw.write('\n')
    fw.close()

if __name__ == '__main__':
    gaussian_mixtures()
