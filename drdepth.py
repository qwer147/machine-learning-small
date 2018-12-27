# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 12:37:56 2018

@author: Administrator
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#数据产生
N = 100
x = np.random.rand(N)*6-3
x.sort()

y = (1+np.cos(x))+np.random.rand(N)*0.04

#print(y)
x = x.reshape(-1,1)  #不知道x的shape属性是多少但是想让x变成只有一列，行数不知道多少

reg = DecisionTreeRegressor(criterion='mse',max_depth=9)
reg.fit(x,y)

#画图
x_test = np.linspace(-3,3,50).reshape(-1,1)
y_pred = reg.predict(x_test)
plt.plot(x,y,linewidth=2,label='真实值')
plt.plot(x_test,y_pred,'b-',label='预测值')
plt.show()

#构造多个模型
depths = [2,4,6,8,10]
clr = 'rgbmy'
models = []
for d in depths:
    reg = DecisionTreeRegressor(criterion='mse',max_depth=d)
    models.append(reg)

plt.plot(x,y,'k^',linewidth=3)
x_test2 = np.linspace(-3,3,50)
#转化后才能画图
x_test2 = x_test2.reshape(-1,1)
for i,model in enumerate(models):
    model.fit(x,y)
    y_pred = model.predict(x_test2)
    lab = '深度为{}的决策树'.format(depths[i])
    plt.plot(x_test2,y_pred,'-',color=clr[i],linewidth=2,label=lab)
plt.legend(loc='best')
plt.grid()
plt.show()
'''
决策树的深度，其中画图通过.linspace返回间隔相等数字画图
注意画图是数据需要转变
'''