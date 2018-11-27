import myKNN
from numpy import *
import operator
from os import listdir
from PIL import Image
import pandas as pda
#图片处理
#先将所有图片转为固定宽高，比如28*28，然后再转为文本
#pillow
for i in range(0, 10):
    for j in range(0, 51):
        im=Image.open("A:/mnist/pics/" + str(i) + "_" + str(j) + ".bmp")
        fh=open("A:/mnist/traindata/" + str(i) + "_" + str(j) + ".txt","a")
        width=im.size[0]# 28
        height=im.size[1]# 28
        for m in range(0,width):
            for k in range(0,height):
                cl=im.getpixel((m,k))
                if(cl==0):
                    fh.write("0")
                else:
                    fh.write("1")
            fh.write("\n")
        fh.close()

#加载数据
def datatoarray(fname):
    arr=[]
    fh=open(fname)
    for i in range(0,28):
        thisline=fh.readline()
        for j in range(0,28):
            arr.append(int(thisline[j]))
    return arr

#建立一个函数取文件名前缀
def seplabel(fname):
    filestr=fname.split(".")[0]
    label=int(filestr.split("_")[0])
    return label

#建立训练数据
def traindata():
    labels=[]
    trainfile=listdir("A:/mnist/traindata")
    num=len(trainfile)
    #长度784（列），每一行存储一个文件
    #用一个数组存储所有训练数据，行：文件总数，列：784
    trainarr=zeros((num,784))
    for i in range(0,num):
        thisfname=trainfile[i]
        thislabel=seplabel(thisfname)
        labels.append(thislabel)
        trainarr[i,:]=datatoarray("A:/mnist/traindata/"+thisfname)
    return trainarr,labels
trainarr,labels=traindata()
xf = pda.DataFrame(trainarr)
yf = pda.DataFrame(labels)
tx2 = xf.as_matrix().astype(int)
ty2 = yf.as_matrix().astype(int)
'''
#用测试数据调用KNN算法去测试，看是否能够准确识别
def datatest():
    trainarr,labels=traindata()
    testlist=listdir("A:/mnist/testdata")
    tnum=len(testlist)
    for i in range(0,tnum):
        thistestfile=testlist[i]
        testarr=datatoarray("A:/mnist/testdata/"+thistestfile)
        rknn=myKNN.KNNClassify(testarr, trainarr, labels, 3)
        print(rknn)
'''

#使用人工神经网络模型
from keras.models import Sequential
from keras.layers.core import Dense,Activation
model=Sequential()
#输入层
model.add(Dense(10,input_dim=len(tx2[0])))
model.add(Activation("relu"))
#输出层
model.add(Dense(1,input_dim=len(ty2[0])))
model.add(Activation("sigmoid"))
#模型的编译
model.compile(loss="mean_squared_error",optimizer="adam")
#训练
model.fit(tx2,ty2,nb_epoch=10,batch_size=10)
#预测分类
rst=model.predict_classes(x).reshape(len(x))
x=0
for i in range(0,len(x2)):
    if(rst[i]!=y[i]):
        x+=1
print(1-x/len(x2))
import numpy as npy
x3=npy.array([[1,-1,-1,1],[1,1,1,1],[-1,1,-1,1]])
rst=model.predict_classes(tx2).reshape(len(tx2))
print(rst)
