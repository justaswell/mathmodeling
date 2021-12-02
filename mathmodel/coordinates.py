import numpy as np
from readtxt import *
import glob as glob


def nor_cal_coor(data_array) -> object:
    a=[[0,0,1300],[5000,0,1700],[0,5000,1700],[5000,5000,1300]]
    #a = [[0, 0, 1200], [5000, 0, 1600], [0, 3000, 1600], [5000, 3000, 1200]]
    a=np.array(a,dtype=int)
    #print(a)
    A=[[2*(a[0][0]-a[1][0]),2*(a[0][1]-a[1][1]),2*(a[0][2]-a[1][2])],
       [2*(a[0][0]-a[2][0]),2*(a[0][1]-a[2][1]),2*(a[0][2]-a[2][2])],
       [2*(a[0][0]-a[3][0]),2*(a[0][1]-a[3][1]),2*(a[0][2]-a[3][2])]]
    lamta=[data_array[1]**2-data_array[0]**2-a[1][0]**2+a[0][0]**2-a[1][1]**2+a[0][1]**2-a[1][2]**2+a[0][2]**2,
           data_array[2]**2-data_array[0]**2-a[2][0]**2+a[0][0]**2-a[2][1]**2+a[0][1]**2-a[2][2]**2+a[0][2]**2,
           data_array[3]**2-data_array[0]**2-a[3][0]**2+a[0][0]**2-a[3][1]**2+a[0][1]**2-a[3][2]**2+a[0][2]**2]
    A=np.array(A,dtype=float)
    lamta=np.array(lamta,dtype=float)
    invA=np.linalg.inv(A)
    X=np.dot(invA,lamta)
    #print(X)
    return X

def diff(A,B):
    return abs(A-B)



def normal_washed():
    filepath="D:/csz/Etopic/fujian3wuganrao.txt"
    data_array=txttoarray(filepath)
    row,_=data_array.shape
    for i in range(row):
        X=nor_cal_coor(data_array[i])
        X=np.array(X,dtype=int)
        print("无干扰坐标",str(i+1),":",X)
    '''correct_coordinates=readtag("D:/csz/Etopic/附件1：UWB数据集/Tag.txt")[:,1:]
    correct_coordinates*=10
    filepath="D:/csz/Etopic/answer/normal/"
    files=glob.glob(filepath+"*")
    coordinates=[]
    difference=[]
    for file in files:
        index=file.split("\\")[-1]
        index=index.split(".")[0]
        #print(index)
        data_array=readtxt(file)
        X=[]
        for data in data_array:
            data=data.split(" ")
            #print(data)
            data=np.array(data,dtype=int)
            #x=nor_cal_coor(data)
            X.append(data)
        X=np.array(X,dtype=int)
        X=np.mean(X,axis=0) #zui xiaoer cheng
        Y=nor_cal_coor(X)
        difference.append(diff(Y,correct_coordinates[int(index)-1]))
        coordinates.append(Y)
    #coordinates=np.array(coordinates,dtype=int)
    difference=np.array(difference,dtype=int)
    print(difference)''' #jisuan you xiao xing
    '''x,y=difference.shape
    n=0
    for i in range(x):
        for j in range(y):
            if(difference[i][j]>=500):
                print(i,j,difference[i][j])
                n+=1
                break
    print(n)''' #ji suan dan wei wucha
    '''XYZ=(difference[:,0]**2+difference[:,1]**2+difference[:,2]**2)**0.5
    mean_XYZ=np.mean(XYZ)
    print(mean_XYZ)

    XY = (difference[:, 0] ** 2 + difference[:, 1] ** 2 ) ** 0.5
    mean_XY = np.mean(XY)
    print(mean_XY)

    X=difference[:,0]
    mean_X = np.mean(X)
    print(mean_X)

    Y=difference[:,1]
    mean_Y = np.mean(Y)
    print(mean_Y)

    Z=difference[:,2]
    mean_Z = np.mean(Z)
    print(mean_Z)'''
    #differences=diff(correct_coordinates,coordinates)
    #print(differences)

def normal_unwashed():
    filepath="D:/csz/Etopic/附件1：UWB数据集/正常数据/109.正常.txt"
    data_array=txttoarray(filepath)
    X=np.mean(data_array,axis=0)
    Y=nor_cal_coor(X)

    '''X=[]
    row,_=data_array.shape
    for i in range(row):
        x=nor_cal_coor(data_array[i])
        X.append(x)
    X = np.array(X, dtype=int)
    X = np.mean(X, axis=0)'''
    print(Y)
    #print(data_array.shape)

if __name__ == '__main__':
    normal_washed()