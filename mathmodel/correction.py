from readtxt import *
from coordinates import nor_cal_coor
from draw import draw3D,draw_distance
import numpy as np


def correcttra(data_array):
    x,y=data_array.shape
    #print(x,y)
    num=0
    for i in range(1,x-1):
        n=data_array[i]-data_array[i-1]
        n1=data_array[i+1]-data_array[i]
        #print(n)
        tag=0
        if(abs(n1[0])>30):
            data_array[i+1,0]=(data_array[i][0]+data_array[i+1][0])/2
            tag=1
        if(abs(n1[1])>30):
            data_array[i+1,1]=(data_array[i][1]+data_array[i+1][1])/2
            tag=1
        if(abs(n1[2])>30):
            data_array[i+1,2]=(data_array[i][2]+data_array[i+1][2])/2
            tag=1
        if(abs(n1[3])>30):
            data_array[i+1,3]=(data_array[i][3]+data_array[i+1][3])/2
            tag=1
        if(tag==1):
            num+=1
    return num,data_array

def difff(data_array):
    x, y = data_array.shape
    for i in range(1, x - 1):
        print(data_array[i]-data_array[i-1])



def main():
    txtpath = "D:/csz/Etopic/附件5：动态轨迹数据.txt"
    data = txttoarray(txtpath)
    data = np.array(data, dtype=int)
    #difff(data)
    OLD=[]


    for i in range(20):
        OLD.append(data[i])
    OLD=np.array(OLD,dtype=int)


    for epoch in range(50):
        num,data=correcttra(data)
        print(num)
    '''if(num<50):
            break'''
    #data[0:20,:]=OLD
    draw_distance(data)
    rows,_=data.shape
    CO=[]
    for i in range(rows):
        x=nor_cal_coor(data[i])
        CO.append(x)
    CO=np.array(CO,dtype=int)
    row,_=CO.shape
    for i in range(1,row-1):
        CO[i][2]=(CO[i-1][2]+CO[i+1][2])/2
    draw3D(CO)




if __name__ == '__main__':
    main()