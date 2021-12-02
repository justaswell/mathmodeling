import numpy as np
import matplotlib.pyplot as plt
from readtxt import *
from mpl_toolkits.mplot3d import Axes3D


def draw_distance(data_array):
    row,line=data_array.shape
    x=range(row)
    #print(x)
    y0=data_array[:,0]
    y1=data_array[:,1]
    y2 = data_array[:, 2]
    y3 = data_array[:, 3]
    #print(y)
    plt.plot(x, y0, 'b*', label="A0")
    plt.plot(x, y1, 'r*', label="A1")
    plt.plot(x, y2, 'g*', label="A2")
    plt.plot(x, y3, 'y*', label="A3")
    '''plt.plot(x, y0, 'b', label="A0")
    plt.plot(x, y1, 'r', label="A1")
    plt.plot(x, y2, 'g', label="A2")
    plt.plot(x, y3, 'y', label="A3")'''
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.title("Fig trajectory")
    plt.legend(loc='lower right')
    plt.ylim(0,7000)
    plt.show()



def draw3D(coordinatesset):
    fig=plt.figure()
    ax=Axes3D(fig)
    x=coordinatesset[:,0]
    y = coordinatesset[:,1]
    z = coordinatesset[:,2]
    ax.scatter3D(x,y,z,cmap='Blues')
    ax.plot3D(x,y,z,'gray')
    plt.show()

def main():
    filepath = "D:/csz/Etopic/附件5：动态轨迹数据.txt"
    data_array = txttoarray(filepath)
    draw_distance(data_array)
    '''filepath = "D:/csz/Etopic/answer/abnormal/24.异常.txt"
    data_array = readtxt(filepath)
    Data = []
    for data in data_array:
        data = data.split(" ")
        # print(data)
        data = np.array(data, dtype=int)
        Data.append(data)
    Data=np.array(Data,dtype=int)
    draw_distance(Data)'''
    '''file_path = "D:/csz/Etopic/附件1：UWB数据集/Tag.txt"
    coordinatesset=readtag(file_path)[:,1:]
    coordinatesset=np.array(coordinatesset,dtype=int)
    print(coordinatesset.shape)
    draw3D(coordinatesset)'''



if __name__ == '__main__':
    main()