 # -*- coding:utf-8 -*-

def compute_acc():
    with open('classify_result','r') as fr:
        count_right = 0
        count = 0
        for line in fr.readlines():
            count += 1
            line = line.strip('\n').split(' ')
            # print(line)
            if line[1] == line[2]:
                count_right += 1
    fr.close()
    print('分类正确的个数：',count_right)
    print('测试集总文档数：',count)
    acc = count_right / count
    print('准确率：',acc)

compute_acc()
