#BP人工神经网络的实现
#1、读取数据
#2、keras.models Sequential   /keras.layers.core Dense Activation
#3、Sequential建立模型
#4、Dense建立层
#5、Activation激活函数
#6、compile模型编译
#7、fit训练（学习）
#8、验证（测试，分类预测）

#使用人工神经网络预测课程销量
#数据的读取与整理
import pandas as pda
fname = "A:/Python_17/lesson.csv"
dataf = pda.read_csv(fname)
x = dataf.iloc[:, 1:5].as_matrix()
y = dataf.iloc[:, 5].as_matrix()

for i  in range(0, len(x)):
    for j in range(0, len(x[i])):
        thisdata = x[i][j]
        if(thisdata == "是" or thisdata =="多" or thisdata == "高"):
            x[i][j] = 1
        else:
            x[i][j] = 0

for i in range(0, len(y)):
    thisdata = y[i]
    if(thisdata == "高"):
        y[i] = 1
    else:
        y[i] = 0

xf = pda.DataFrame(x)
yf = pda.DataFrame(y)
x2 = xf.as_matrix().astype(int)
y2 = yf.as_matrix().astype(int)

# 使用ANN
from keras.models import Sequential
from keras.layers.core import Dense,Activation
model = Sequential()

# 输入层
model.add(Dense(10, input_dim = len(x2[0])))
model.add(Activation("relu"))

# 输出层
model.add(Dense(1, input_dim = len(y2[0])))
model.add(Activation("sigmoid"))

# 模型的编译
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练
model.fit(x2, y2, nb_epoch = 1000, batch_size = 100)

# 预测分类
rst=model.predict_classes(x2).reshape(len(x2))
x=0
for i in range(0,len(x2)):
    if(rst[i]!=y[i]):
        x+=1
print(1-x/len(x2))

import numpy as npy
x3=npy.array([[1, -1, -1, 1], [1, 1, 1, 1], [-1, 1, -1, 1]])
rst=model.predict_classes(x3).reshape(len(x3))
print("其中第一门课的预测结果为:"+str(rst[0]))















