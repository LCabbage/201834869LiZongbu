# coding = utf-8
from sklearn.externals import joblib
from sklearn.cluster import DBSCAN
from sklearn import metrics
import time

def Dbscan():
    weight = joblib.load('result/weight.pkl')
    # print(weight)
    fw = open('result/result.txt', 'a', encoding='utf-8')
    fw.write('DBSCAN')
    fw.write('\n')

    clf = DBSCAN()
    time_start = time.time()
    s = clf.fit(weight)
    time_run = time.time() - time_start
    print(s)

    # 核心点的索引，因为labels_不能区分核心点还是边界点，所以需要用这个索引确定核心点
    print(clf.core_sample_indices_)
    # 训练样本的核心点
    print(clf.components_)

    # 每个样本所属的类别
    labels = clf.labels_
    print(labels)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

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
    Dbscan()
