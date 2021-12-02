import numpy as np
import glob as glob
import random
#from washdata import writetotxt

def readtxt(file_path):
    file_name_list=[]
    with open(file_path, encoding='utf-8') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    #hairandom.shuffle(file_name_list)
    return file_name_list

def txttoarray(file_path):
    filenamelist=readtxt(file_path)[1:]
    #print(len(filenamelist))
    filedata=[]
    for i in range(0,len(filenamelist),4):
        data=[]
        for j in range(4):
            linedata=filenamelist[i+j].split(':')[5:]
            data.append(linedata)
        #data=np.array(data,dtype=float)
        #print(data.shape)
        if(data[0][2]==data[1][2]==data[2][2]==data[3][2] and data[0][3]==data[1][3]==data[2][3]==data[3][3]):#检查缺失值
            temp=[]
            temp.append(data[0][0])
            temp.append(data[1][0])
            temp.append(data[2][0])
            temp.append(data[3][0])
            temp=np.array(temp,dtype=int)
            filedata.append(temp)
        else:
            print("wrong data!")
            print(filenamelist[i])
            break

    filedata=np.array(filedata,dtype=int)

    return filedata

'''def solve_func(unsolved_value):
    x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
    return [
        x ** 2 + y ** 2 +(z-130)**2 - 76**2,
        (x-500) ** 2 + y ** 2 +(z-170)**2 -455**2,
        (x-500) ** 2 + (y-500) ** 2 +(z-130)**2 -455**2,
        #x ** 2 + (y-5000) ** 2 +(z-1700)**2 -6300**2,
    ]
'''
def writetotxt(data_array,txt_path,txt_name):
    f = open(txt_path + txt_name, 'w')
    x,y=data_array.shape
    for i in range(x):
        for j in range(y):
            f.write(str(data_array[i][j]) + " ")
        f.write("\n")
    f.close()

def readtag(data_path):
    tags=readtxt(data_path)[2:]
    print(len(tags))
    Tag=[]
    for tag in tags:
        tag=' '.join(tag.split())
        tag=tag.split(" ")
        Tag.append(tag)
    Tag=np.array(Tag,dtype=int)
    return Tag


def main():
    file_path = "D:/csz/Etopic/answer/abnormal/1.异常.txt"
    print(readtxt(file_path))
    '''filename="dataset.txt"
    files=glob.glob(file_path+'*')
    print(len(files))
    dataset=[]
    for file in files:
        Data=readtxt(file)
        for data in Data:
            data=data.split(" ")
            #data=data.reshape(-1,4)
            print(data)
            data=np.array(data,dtype=int)
            dataset.append(data)
    print(len(dataset))
    dataset=np.array(dataset,dtype=int)
    print(dataset.shape)
    writetotxt(dataset,file_path,filename)'''
    '''data=txttoarray(file_path)
    data=data.reshape(-1,4)
    print(data)'''


if __name__ == '__main__':
    main()