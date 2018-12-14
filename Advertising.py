# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 23:22:49 2018

@author: Administrator
"""
#加载相应的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#数据存放的位置,注意'\'要变成'/'
'''
path = 'D:alldata\//回归案例数据.Advertising.csv'
data = pd.read_csv(path)
'''
#由于路径中存在中文读取会报错，改成如下则可以
f = open('D:/alldata/回归案例数据/Advertising.csv')
data = pd.read_csv(f,index_col=0)
#读取数据时候有'Unnamed: 0'列，index_col=0第一列做行索引
#data = data.drop('Unnamed: 0',axis=1)
data.head()

#划分特征与y
X = data[['TV','Radio','Newspaper']]
y = data[['Sales']]

#画图
plt.figure(1)
plt.subplot(111)
plt.plot(data['TV'],y,'ro',label='TV')
plt.plot(data['Radio'],y,'g^',label='Radio')
plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
plt.legend(loc='lower right')
plt.grid()
plt.show()


'''
#每个一张
plt.subplot()
plt.subplot(311)
plt.plot(data['TV'],y,'ro',label='TV')
plt.subplot(312)
plt.plot(data['Radio'],y,'g^',label='Radio')
plt.subplot(313)
plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
plt.grid()
plt.show()
'''

'''
#矩阵散点图
from pandas.plotting import scatter_matrix
scatter_matrix(data,diagonal='kde',alpha=.3)
plt.show()
'''
#数据划分
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg = LinearRegression()
model = linreg.fit(x_train, y_train)
print (model)
#相关系数
print (linreg.coef_)
#截距项
print (linreg.intercept_)

#均方误差
pred = linreg.predict(x_test)
mse = mean_squared_error(pred,y_test)
print('均方误差是%f'%mse)
print('均方根误差是%f'%np.sqrt(mse))
'''
#手写均方误差
mse2 = np.average((np.array(pred) - np.array(y_test)) ** 2)
print('均方误差是%f'%mse2)
print('均方根误差是%f'%np.sqrt(mse2))

'''
#画图看拟合效果
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, pred, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()