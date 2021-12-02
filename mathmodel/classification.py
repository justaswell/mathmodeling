import random
import pandas as pd
import numpy as np
from readtxt import *
from coordinates import nor_cal_coor
from draw import draw3D
from abnormal_coordinates import cal_ab

# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k,1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids


# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(dataSet, k)

    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    n=0
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)
        n+=1
        print("已迭代",str(n),"次!")

    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])

    return centroids, cluster


def judgeisabnormal(centroids,data):
    data=nor_cal_coor(data)
    diff = np.tile(data, (2, 1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
    squaredDiff = diff ** 2  # 平方
    squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
    distance = squaredDist ** 0.5  # 开根号
    #print(distance)
    if(distance[0]>distance[1]):
        return 1
    else:
        return 0


if __name__ == '__main__':
    txtpath = "D:/csz/Etopic/fujian4.txt"
    '''filepath = "D:/csz/Etopic/answer/classification.txt"
    
    data_array = readtxt(filepath)
    Data = []
    for data in data_array:
        data = data.split(" ")
        # print(data)
        data = np.array(data, dtype=int)
        Data.append(data)
    Data = np.array(Data, dtype=int)
    dataset=[]
    row,_=Data.shape
    for i in range(row):
        coor=nor_cal_coor(Data[i])
        dataset.append(coor)
    centroids, cluster = kmeans(dataset, 2)
    print('质心为：%s' % centroids)
    #print('集群为：%s' % cluster)
    centroids=np.array(centroids,dtype=float)'''
    centroids=[[2282.0235548583414, 2521.1936792165634, -201.3562107815174], [2660.9297508120276, 2506.0091000078405, 3036.4733018138772]]
    centroids=np.array(centroids,dtype=int)
    #data=[1280,4550,4550,6300]
    data=txttoarray(txtpath)
    data=np.array(data,dtype=int)
    print(data.shape)
    row,_=data.shape
    DATA=[]
    for i in range(row):
        if(judgeisabnormal(centroids,data[i])):
            x=nor_cal_coor(data[i])
            print(x)
            DATA.append(x)
        else:
            y=data[i].reshape(-1,4)
            #print(y)
            x=cal_ab(y)
            DATA.append(x)
            print("abnormal!")
    DATA=np.array(DATA,dtype=int)
    row,_=DATA.shape
    for i in range(1,row-1):
        DATA[i][2]=(DATA[i-1][2]+DATA[i+1][2])/2
    draw3D(DATA)


