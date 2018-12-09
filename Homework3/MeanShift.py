# coding = utf-8
from sklearn.externals import joblib
from sklearn.cluster import MeanShift,estimate_bandwidth
from sklearn import metrics
import time
import numpy as np

def mean_shift():
    weight = joblib.load('result/weight.pkl')
    # print(weight)
    fw = open('result/result.txt', 'a', encoding='utf-8')
    fw.write('MeanShift')
    fw.write('\n')

    #estimate_bandwidth()用于生成mean - shift窗口的尺寸，其参数的意义为：从数据中随机选取(n_samples=500)500个样本，计算每一对样本的距离，然后选取这些距离的(quantile=0.2)0.2分位数作为返回值，
    bandwidth = estimate_bandwidth(weight, quantile=0.2)
    print(bandwidth)
    # bin_seeding設置為True就不會把所有的點初始化為核心位置，從而加速算法
    clf = MeanShift(bandwidth=bandwidth,bin_seeding=True)
    time_start = time.time()
    s = clf.fit(weight)
    time_run = time.time() - time_start
    print(s)  # MiniBatchKMeans(batch_size=100, ...n_clusters=89, 批处理默为100
    # 中心点
    print('中心点：')  # print(len(clf.cluster_centers_[0])) #5097（维度） # print(len(clf.cluster_centers_)) # 89 （89个簇）
    print(clf.cluster_centers_)  # 聚类中心均值向量矩阵
    print(len(clf.cluster_centers_))
    # 每个文档的预测标签
    print(clf.labels_)

    pred_label = []  # 存储2742个文本的预测标签
    a = {}  # key:类别标签,value:此类的所有文档
    i = 1
    while i <= len(clf.labels_):
        # print(i, clf.labels_[i - 1])  # 第几个文本，对应的类别标签
        pred_label.append(clf.labels_[i - 1])

        if clf.labels_[i - 1] not in a.keys():
            a[clf.labels_[i - 1]] = []
        else:
            a[clf.labels_[i - 1]].append(i)

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
    nmi = metrics.normalized_mutual_info_score(true_lable, pred_label)
    print('MeanShift的NMI值为:%f' % ( nmi))  # 结果越相似NMI值应接近1；算法结果很差则NMI值接近0
    print('运行时间:', time_run)

    fw.write('\t')
    fw.write('NMI值为:' + str(nmi) + ' ' + '运行时间:' + str(time_run))

    localtime = time.localtime(time.time())  # time.localtime()方法，作用是格式化时间戳为本地的时间
    time_format = time.strftime('%Y-%m-%d %H:%M:%S', localtime)  # 格式化制定形式
    fw.write(' ' + '本地时间：' + time_format)
    fw.write('\n')
    fw.close()
if __name__ == '__main__':
    mean_shift()
