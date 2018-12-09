# coding = utf-8
from sklearn.externals import joblib
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def spectral_clustering():
    weight = joblib.load('result/weight.pkl')
    # print(weight)
    fw = open('result/result.txt', 'a', encoding='utf-8')
    fw.write('SpectralClustering')
    fw.write('\n')

    num = 89
    clf = SpectralClustering(n_clusters=num)
    time_start = time.time()
    s = clf.fit(weight)
    time_run = time.time() - time_start
    print(s)

    print(clf.affinity_matrix_)# 亲和度矩阵

    # 每个样本所属的类别
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
    print('聚类数为%d的NMI值为:%f' % (num, nmi))  # 结果越相似NMI值应接近1；算法结果很差则NMI值接近0
    print('运行时间:', time_run)

    fw.write('\t')
    fw.write('NMI值为:' + str(nmi) + ' ' + '运行时间:' + str(time_run))

    localtime = time.localtime(time.time())  # time.localtime()方法，作用是格式化时间戳为本地的时间
    time_format = time.strftime('%Y-%m-%d %H:%M:%S', localtime)  # 格式化制定形式
    fw.write(' ' + '本地时间：' + time_format)
    fw.write('\n')
    fw.close()

    ####  降维 ####
    pca = PCA(n_components=2)  # 降维两维
    print('pca:', pca)
    new_weight = pca.fit_transform(weight)  # 重新计算成二维形式
    print('newWeight:', new_weight)

    # 绘制散点图（scatter），横轴为x，获取的第1列数据；纵轴为y，获取的第2列数据；c=y_pred对聚类的预测结果画出散点图，marker='o'说明用点表示图形。
    plt.scatter(new_weight[:, 0], new_weight[:, 1], c=pred_label, marker='o')
    plt.title('SpectralClustering')
    plt.show()

if __name__ == '__main__':
    spectral_clustering()
