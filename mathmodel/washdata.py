from readtxt import txttoarray
import numpy as np
import os
import glob as glob

def isabnormal(data_array):
    x,y=data_array.shape
    #print(x)
    mean=np.mean(data_array,axis=0)
    std=np.std(data_array,axis=0)
    #print(mean)
    #print(std)
    sup=mean+3*std
    inf=mean-3*std
    sup=np.array(sup,dtype=int)
    inf = np.array(inf, dtype=int)
    #print(sup)
    #print(inf)
    line=[]
    for i in range(x):
        if(data_array[i][0]>sup[0] or data_array[i][0]<inf[0]):
            line.append(i)
            print(str(i)+" line is abnormal!")
            continue
        if (data_array[i][1] > sup[1] or data_array[i][1] < inf[1]):
            line.append(i)
            print(str(i)+" line is abnormal!")
            continue
        if (data_array[i][2] > sup[2] or data_array[i][2] < inf[2]):
            line.append(i)
            print(str(i)+" line is abnormal!")
            continue
        if (data_array[i][3] > sup[3] or data_array[i][3] < inf[3]):
            line.append(i)
            print(str(i)+" line is abnormal!")
            continue
    print(data_array.shape)
    line=np.array(line,dtype=int)
    data_array=np.delete(data_array, line, axis=0)
    print(data_array.shape)
    return data_array



def isrepeat(data_array):
    print(data_array.shape)
    data_array=np.unique(data_array,axis=0)
    print(data_array.shape)
    return data_array

def issimilar(data_array):
    x,_=data_array.shape
    line=[]
    for i in range(x):
        for j in range(i+1,x):
            cos=(data_array[i][0]*data_array[j][0]+data_array[i][1]*data_array[j][1]+data_array[i][2]*data_array[j][2]+data_array[i][3]*data_array[j][3])/\
                (((data_array[i][0]*data_array[i][0]+data_array[i][1]*data_array[i][1]+data_array[i][2]*data_array[i][2]+data_array[i][3]*data_array[i][3])**0.5)*(
                (data_array[j][0]*data_array[j][0]+data_array[j][1]*data_array[j][1]+data_array[j][2]*data_array[j][2]+data_array[j][3]*data_array[j][3])**0.5))
            if(cos>=0.9999):
                print("cos:",cos)
                print("data ",str(i)," and data ",str(j)," is similar!")
                if j not in line:
                    line.append(j)
    return line


def writetotxt(data_array,txt_path,txt_name):
    f = open(txt_path + txt_name, 'w')
    x,y=data_array.shape
    for i in range(x):
        for j in range(y):
            f.write(str(data_array[i][j]) + " ")
        f.write("\n")
    f.close()



def main():
    file_path="D:/csz/Etopic/附件1：UWB数据集/正常数据/109.正常.txt"
    txt_name="109.正常.txt"
    txt_path="D:/csz/Etopic/NN/"
    data=txttoarray(file_path)
    data=isabnormal(data)
    data=isrepeat(data)
    data=np.delete(data,issimilar(data),axis=0)
    writetotxt(data,txt_path,txt_name)
    ''' file_path = "D:/csz/Etopic/附件1：UWB数据集/异常数据/"
    # = "D:/csz/Etopic/附件1：UWB数据集/z常数据/100.异常.txt"
    files = glob.glob(file_path + '*')

    txt_path="D:/csz/Etopic/answer/abnormal/"
    txt_name="classification1.txt"
    X=[]
    #n=0
    for file in files:
        txt_name=file.split("\\")[-1]
        #print(txt_name)
        data=txttoarray(file)
        #data = txttoarray(file_path)
        data=isabnormal(data)
        data=isrepeat(data)
        #line=issimilar(data)
        #n+=len(line)
        #data=np.delete(data,line,axis=0)
        #row,y=data.shape
        #b=np.ones(row)
        #data=np.insert(data,y,b,axis=1)
        #data=np.array(data,dtype=int)
        #row,_=data.shape
        #for i in range(row):
        #X.append(data)
        writetotxt(data,txt_path,txt_name)
    #print(X)
    #X=np.array(X,dtype=int)
    #'''



if __name__ == '__main__':
    main()