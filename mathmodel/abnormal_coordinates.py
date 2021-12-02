from readtxt import *
import numpy as np
from coordinates import nor_cal_coor
import math
import glob as glob


def correction(data_array):
    a0 = data_array[:, 0]
    a1 = data_array[:, 1]
    a2 = data_array[:, 2]
    a3 = data_array[:, 3]
    mean0 = (np.max(a0) + np.min(a0)) / 2
    mean1 = (np.max(a1) + np.min(a1)) / 2
    mean2 = (np.max(a2) + np.min(a2)) / 2
    mean3 = (np.max(a3) + np.min(a3)) / 2
    a0[a0 > mean0] = 0
    a1[a1 > mean1] = 0
    a2[a2 > mean2] = 0
    a3[a3 > mean3] = 0
    a0 = a0[a0 != 0]
    a1 = a1[a1 != 0]
    a2 = a2[a2 != 0]
    a3 = a3[a3 != 0]
    #print(a0.shape)
    #print(a1.shape)
    #print(a2.shape)
    #print(a3.shape)
    x0 = np.mean(a0)
    x1 = np.mean(a1)
    x2 = np.mean(a2)
    x3 = np.mean(a3)
    X=[x0,x1,x2,x3]
    X=np.array(X,dtype=int)
    #print(X)
    return X


def abnor_cal_coor(r,i):
    a = [[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]]
    #a = [[0, 0, 1200], [5000, 0, 1600], [0, 3000, 1600], [5000, 3000, 1200]]
    a=np.array(a,dtype=int)
    r=np.delete(r,i,axis=0)
    a=np.delete(a,i,axis=0)
    #print(r)
    #print(a)
    A1 = r[0] ** 2 - a[0][0] ** 2 - a[0][1] ** 2 - a[0][2] ** 2
    A2 = r[1] ** 2 - a[1][0] ** 2 - a[1][1] ** 2 - a[1][2] ** 2
    A3 = r[2] ** 2 - a[2][0] ** 2 - a[2][1] ** 2 - a[2][2] ** 2
    A21=-(A2-A1)/2
    A31=-(A3-A1)/2
    X21 = a[1][0] - a[0][0]
    X31 = a[2][0] - a[0][0]
    Y21 = a[1][1] - a[0][1]
    Y31 = a[2][1] - a[0][1]
    Z21 = a[1][2] - a[0][2]
    Z31 = a[2][2] - a[0][2]
    D=X21*Y31-Y21*X31
    #print(X21,X31,Y21,Y31)
    #print(D)
    B0=(A21*Y31-A31*Y21)/D
    B1=(Y21*Z31-Y31*Z21)/D
    C0=(A31*X21-A21*X31)/D
    C1=(X31*Z21-X21*Z31)/D
    E=B1**2+C1**2+1
    F=B1*(B0-a[0][0])+C1*(C0-a[0][1])-a[0][2]
    G=(B0-a[0][0])**2+(C0-a[0][1])**2+a[0][2]**2-r[0]**2
    #print(F**2-E*G)
    if((F**2-E*G)>=0):
        z1=(-F+math.sqrt(F**2-E*G))/E
        z2=(-F-math.sqrt(F**2-E*G))/E
    else:
        z1 = (-F + math.sqrt(  E * G-F ** 2)) / E
        z2 = (-F - math.sqrt(E * G-F ** 2)) / E
        #print("some errors!")
    x1=B0+B1*z1
    y1=C0+C1*z1
    x2 = B0 + B1 * z2
    y2 = C0 + C1 * z2
    #print(x1,y1,z1)
    #print(x2,y2,z2)
    X=[[x1,y1,z1],[x2,y2,z2]]
    X=np.array(X,dtype=int)
    return X

def cal_distance(X,i):
    a = [[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]]
    #a = [[0, 0, 1200], [5000, 0, 1600], [0, 3000, 1600], [5000, 3000, 1200]]
    dis1=math.sqrt((X[0][0]-a[i][0])**2+(X[0][1]-a[i][1])**2+(X[0][2]-a[i][2])**2)
    dis2 = math.sqrt((X[1][0] - a[i][0]) ** 2 + (X[1][1] - a[i][1]) ** 2 + (X[1][2] - a[i][2]) ** 2)
    return dis1,dis2

def cal_r(X):
    a = [[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]]
    #a = [[0, 0, 1200], [5000, 0, 1600], [0, 3000, 1600], [5000, 3000, 1200]]
    dis0= math.sqrt((X[0]-a[0][0])**2+(X[1]-a[0][1])**2+(X[2]-a[0][2])**2)
    dis1= math.sqrt((X[0]-a[1][0])**2+(X[1]-a[1][1])**2+(X[2]-a[1][2])**2)
    dis2 =math.sqrt((X[0]-a[2][0])**2+(X[1]-a[2][1])**2+(X[2]-a[2][2])**2)
    dis3 =math.sqrt((X[0]-a[3][0])**2+(X[1]-a[3][1])**2+(X[2]-a[3][2])**2)
    r=[dis0,dis1,dis2,dis3]
    r=np.array(r,dtype=float)
    return r

def difff(A,B):
    return abs(A-B)


def cal_difference(X,Y):
    std=math.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2+(X[2]-Y[2])**2)
    return std

def cal_dis_diff(X,Y,i):
    X=np.delete(X,i,axis=0)
    Y = np.delete(Y, i, axis=0)
    #print(X.shape,Y.shape)
    std=math.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2+(X[2]-Y[2])**2)
    return std


def mean_error(data_array):
    max=np.max(data_array,axis=0)
    min=np.min(data_array,axis=0)
    return max-min

def islegal(ab_nor,Dis):
    x,y,z=ab_nor.shape
    #dis=Dis
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if( ab_nor[i][j][k]<0):
                    Dis[i][j]=0
                    continue
    return Dis

def cal_ab(Data):
    row, _ = Data.shape
    # print(Data.shape)
    for k in range(row):
        r = Data[k]
        # print(r)
        # r=np.array(r,dtype=int)
        Dis = np.ones((4, 2))
        ab_nor = []
        diff = []
        for i in range(4):

            X = abnor_cal_coor(r, i)
            # print(X[0],X[1])
            dis1, dis2 = cal_distance(X, i)
            if (dis1 - r[i] > 0):
                Dis[i][0] = 0
            if (dis2 - r[i] > 0):
                Dis[i][1] = 0

            temp = r[i]
            r[i] = dis1
            y1 = nor_cal_coor(r)
            # print(y1)
            # y1=cal_r(y1)
            # print(y1,R)
            # Y1=cal_dis_diff(R,y1,i)
            r[i] = dis2
            y2 = nor_cal_coor(r)
            ab_nor.append([y1, y2])
            # print(y2)
            # y2=cal_r(y2)
            # print(y2,R)
            # Y2 = cal_dis_diff(R, y2, i)
            # print(Y1,Y2)
            # print(cal_difference(X[0],Y1),cal_difference(X[1],Y2))
            # print(ab_nor)
            diff.append([cal_difference(X[0], y1), cal_difference(X[1], y2)])
            # print(cal_r(Y1),cal_r(Y2))
            r[i] = temp
        # Dis=np.array(Dis,dtype=int)
        ab_nor = np.array(ab_nor, dtype=int)
        # print(ab_nor)
        diff = np.array(diff, dtype=float)
        # print(Dis)
        ##print(ab_nor)
        Dis = islegal(ab_nor, Dis)
        final = Dis * diff
        #print(final)
        min = [1e10, 0, 0]
        for i in range(4):
            for j in range(2):
                # print(final[i][j])
                if (final[i][j] != 0 and final[i][j] < min[0]):
                    min = [final[i][j], i, j]
        print("干扰coordinates", k + 1, ":", ab_nor[min[1]][min[2]])
        a=ab_nor[min[1]][min[2]]
    return a


def main():
    filepath="D:/csz/Etopic/fujian4.txt"
    y=txttoarray(filepath)
    y=np.array(y,dtype=int)
    print(y.shape)
    cal_ab(y)

    '''filepath = "D:/csz/Etopic/answer/abnormal/1.异常.txt"
    data_array = readtxt(filepath)
    Data = []
    for data in data_array:
        data = data.split(" ")
        # print(data)
        data = np.array(data, dtype=int)
        Data.append(data)
    Data = np.array(Data, dtype=int)'''
    '''correct_coordinates = readtag("D:/csz/Etopic/附件1：UWB数据集/Tag.txt")[:, 1:]
    correct_coordinates *= 10
    filepath="D:/csz/Etopic/answer/abnormal/"
    files=glob.glob(filepath+"*")
    differences=[]
    for file in files:
        index = file.split("\\")[-1]
        index = index.split(".")[0]
        index=int(index)
        #print(index)
        Data=readtxt(file)
        Y = []
        for data in Data:
            data = data.split(" ")
            # print(data)
            data = np.array(data, dtype=int)
            # x=nor_cal_coor(data)
            Y.append(data)
        Data = np.array(Y, dtype=int)
        row,_=Data.shape
        #print(Data.shape)
        x=[]
        for k in range(row):
            r=Data[k]
            #print(r)
            #r=np.array(r,dtype=int)
            Dis=np.ones((4,2))
            ab_nor=[]
            diff=[]
            for i in range(4):

                X=abnor_cal_coor(r,i)
                #print(X[0],X[1])
                dis1,dis2=cal_distance(X,i)
                if(dis1-r[i]>0):
                    Dis[i][0]=0
                if(dis2-r[i]>0):
                    Dis[i][1]=0

                temp=r[i]
                r[i]=dis1
                y1=nor_cal_coor(r)
                #print(y1)
                #y1=cal_r(y1)
                #print(y1,R)
                #Y1=cal_dis_diff(R,y1,i)
                r[i]=dis2
                y2=nor_cal_coor(r)
                ab_nor.append([y1,y2])
                #print(y2)
                #y2=cal_r(y2)
                #print(y2,R)
                #Y2 = cal_dis_diff(R, y2, i)
                #print(Y1,Y2)
                #print(cal_difference(X[0],Y1),cal_difference(X[1],Y2))
                #print(ab_nor)
                diff.append([cal_difference(X[0],y1),cal_difference(X[1],y2)])
                #print(cal_r(Y1),cal_r(Y2))
                r[i]=temp
            #Dis=np.array(Dis,dtype=int)
            ab_nor=np.array(ab_nor,dtype=int)
            #print(ab_nor)
            diff=np.array(diff,dtype=float)
            #print(Dis)
            ##print(ab_nor)
            Dis=islegal(ab_nor,Dis)
            final=Dis*diff
            #print(final)
            min=[1e10,0,0]
            for i in range(4):
                for j in range(2):
                    #print(final[i][j])
                    if(final[i][j]!=0 and final[i][j]<min[0]):
                        min=[final[i][j],i,j]
            #print("干扰coordinates",k+1,":",ab_nor[min[1]][min[2]])
            #print(min)
            x.append(ab_nor[min[1]][min[2]])
        x=np.array(x,dtype=int)
        #print(x)
        x=np.mean(x,axis=0)
        #print(x)
        difference=difff(x,correct_coordinates[index-1])
        #print(difference)
        differences.append(difference)
    differences=np.array(differences,dtype=int)'''

    #print(differences)
    '''xd, yd = differences.shape
    n = 0
    for i in range(xd):
        for j in range(yd):
            if (differences[i][j] >= 500):
                #print(i, j, differences[i][j])
                n += 1
                break
    print(n)'''
    '''XYZ = (differences[:, 0] ** 2 + differences[:, 1] ** 2 + differences[:, 2] ** 2) ** 0.5
    mean_XYZ = np.mean(XYZ)
    print(mean_XYZ)

    XY = (differences[:, 0] ** 2 + differences[:, 1] ** 2) ** 0.5
    mean_XY = np.mean(XY)
    print(mean_XY)

    X = differences[:, 0]
    mean_X = np.mean(X)
    print(mean_X)

    Y = differences[:, 1]
    mean_Y = np.mean(Y)
    print(mean_Y)

    Z = differences[:, 2]
    mean_Z = np.mean(Z)
    print(mean_Z)'''
        #print(diff*Dis)
        #print(r)
    '''for i in range(row):
        r=Data[i]
        r.reshape(-1,1)
        Dis=[]
        for i in range(4):
            X=abnor_cal_coor(r,i)
            #print(X)
            dis1,dis2=cal_distance(X,i)
            Dis.append([dis1,dis2])
        Dis=np.array(Dis,dtype=int)
        print(Dis)'''
        #r=r.reshape(4,1)
        #print(r)
        #err=Dis-r
        #print(err)


if __name__ == '__main__':
    main()