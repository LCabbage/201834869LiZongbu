# coding = utf-8
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn import metrics
import time

def K_Means():
    weight = joblib.load('result/weight.pkl')
    # print(weight)
    fw = open('result/result.txt', 'a', encoding='utf-8')
    fw.write('KMeans')
    fw.write('\n')
    for num in range(85, 90):
        print('聚类数为：', num)
        clf = KMeans(n_clusters=num)
        time_start = time.time()
        s = clf.fit(weight)
        time_run = time.time() - time_start
        print(s)  # KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, n_clusters=89, n_init=10,...)

        # 中心点
        print('中心点：')  # print(len(clf.cluster_centers_[0])) #5097（维度） # print(len(clf.cluster_centers_)) # 89 （89个簇）
        print(clf.cluster_centers_)  # 聚类中心均值向量矩阵

        # 每个样本所属的簇[ 0 55  1 ... 11 80 10]  2472个样本聚类结果
        print(clf.labels_)  # print(clf.predict(weight)) #这两个输出一样  # print(clf.fit_predict(weight)) #返回各自文本的所被分配到的类索引

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
        fw.write('聚类数为' + str(num) + '的NMI值为:' + str(nmi) + ' ' + '运行时间:' + str(time_run))

        localtime = time.localtime(time.time()) # time.localtime()方法，作用是格式化时间戳为本地的时间
        time_format = time.strftime('%Y-%m-%d %H:%M:%S',localtime)  # 格式化制定形式
        fw.write(' ' + '本地时间：'+ time_format)
        fw.write('\n')
    fw.close()

    # #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数  1878.470663247612
    # print('clf.inertia_:',clf.inertia_)




if __name__ == '__main__':
    K_Means()

