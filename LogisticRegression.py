# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:30:50 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#为了方便处理数据
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.datasets import load_iris

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
#数据
iris = load_iris()
x,y = iris.data,iris.target
#采取前两列
x = x[:,:2]


'''
    x = StandardScaler().fit_transform(x)
    lr = LogisticRegression()   # Logistic回归模型
    lr.fit(x, y.ravel())   
'''
#以上代码可以简化为一下,('sc',StandardScaler()sc表示命名
lr = Pipeline([('sc',StandardScaler()),('clf',LogisticRegression())])
lr.fit(x,y)
pred = lr.predict(x)

'''
lr2 = make_pipeline(StandardScaler(),LogisticRegression())
lr2.fit(x,y)
'''

#训练集上的准确率
y1 = y.reshape(-1)
#布尔类型
result = y1==pred
print('在训练集上的准确率为%f'%np.mean(result))
#混淆矩阵
confusion = confusion_matrix(pred,y)
'''
    0   1   2       预测
  0
  1
  2  
  真实
'''
#混淆矩阵图示化
classes = list(set(y))
classes.sort()
plt.imshow(confusion,cmap=plt.cm.Blues)
#49行到62行是为了设置坐标轴以及文本
#indices是index的复数
indices = range(len(confusion))
#设置坐标刻度文本
plt.xticks(indices,classes)
plt.yticks(indices,classes)
plt.colorbar()
plt.xlabel('pred')
plt.ylabel('y')
#添加文本
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index,second_index,confusion[first_index][second_index])
plt.show()


#图示化
#采样数量
N,M=50,50
x1_min,x1_max = x[:,0].min(),x[:,0].max()
x2_min,x2_max = x[:,1].min(),x[:,1].max()
#np.linspace()等差
t1 = np.linspace(x1_min,x1_max,N)
t2 = np.linspace(x2_min,x2_max,M)
#x1为50*50,相当于t1装置后重复50次，x2同理
x1, x2 = np.meshgrid(t1, t2)
#x1.flat迭代器2500=50*50
x_test = np.stack((x1.flat, x2.flat), axis=1)
#颜色
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

#预测
y_hat = lr.predict(x_test)
#预测值列转化成50*50
y_hat = y_hat.reshape(x1.shape)
#plt.pcolormesh()会根据y_predict自动在cmap里面选择颜色
plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
#样本展示
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()
plt.show()