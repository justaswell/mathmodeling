from readtxt import *
import glob as glob
from washdata import writetotxt


def data_list(file_path):
    files=glob.glob(file_path+'*')
    datalist=[]
    for file in files:
        data_array = readtxt(file)
        X = []
        for data in data_array:
            data = data.split(" ")
            data = np.array(data, dtype=int)
            X.append(data)
        X = np.array(X, dtype=int)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        datalist.append([mean,std])
    datalist=np.array(datalist,dtype=int)
    return datalist

def find_error(data,datalist):
    x,_,_=datalist.shape
    for i in range(x):
        inf=datalist[i][0]-3*datalist[i][1]
        sup=datalist[i][0]+3*datalist[i][1]
        for j in range(4):
            n=0
            if (data[0] > inf[0] and data[0] < sup[0]):
                n += 1
            if (data[1] > inf[1] and data[1] < sup[1]):
                n += 1
            if (data[2] > inf[2] and data[2] < sup[2]):
                n += 1
            if (data[3] > inf[3] and data[3] < sup[3]):
                n += 1
            if(n>=3):
                return datalist[i]

def main():
    file_path="D:/csz/Etopic/answer/normal/"
    datalist=data_list(file_path)
    print(datalist[108,:,:])
    #writetotxt(datalist[:,:,:],"D:/csz/Etopic/","mean.txt")
    data=[2230,3230,4910,5180]
    data=np.array(data,dtype=int)
    print(find_error(data,datalist))


if __name__ == '__main__':
    main()