# coding = utf-8
from sklearn.externals import joblib
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import time
import numpy as np

def AP():
    weight = joblib.load('result/weight.pkl')
    # print(weight)
    fw = open('result/result.txt', 'a', encoding='utf-8')
    fw.write('AffinityPropagation')
    fw.write('\n')

    clf = AffinityPropagation()
    time_start = time.time()
    s = clf.fit(weight)
    time_run = time.time() - time_start
    print(s)  # AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,damping=0.5, max_iter=200, preference=None, verbose=False)
    # 中心点
    print('中心点：')
    print(clf.cluster_centers_)  # Cluster centers
    print(clf.cluster_centers_indices_)  # cluster_centers_indices_ : 存放聚类中心的数组  KMeans，MiniBatchKmeans没有此属性
    print('n_clusters_:',len(clf.cluster_centers_indices_))  # 聚了多少类

    # labels_ :存放每个点的分类的数组
    print(clf.labels_)
    # print(np.unique(clf.labels_))  # 89个类的数组形式，用来计算聚了几个类  和上面len(clf.cluster_centers_indices_)功能差不多

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
    print('AffinityPropagation的NMI值为:%f' % (nmi))  # 结果越相似NMI值应接近1；算法结果很差则NMI值接近0
    print('运行时间:', time_run)

    fw.write('\t')
    fw.write('NMI值为:' + str(nmi) + ' ' + '运行时间:' + str(time_run))

    localtime = time.localtime(time.time())  # time.localtime()方法，作用是格式化时间戳为本地的时间
    time_format = time.strftime('%Y-%m-%d %H:%M:%S', localtime)  # 格式化制定形式
    fw.write(' ' + '本地时间：' + time_format)
    fw.write('\n')
    fw.close()

if __name__ == '__main__':
    AP()
