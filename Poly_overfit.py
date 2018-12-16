# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:22:35 2018

@author: Administrator
"""

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
#多项式
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

#作图中文显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#产生数据(加噪音)
np.random.seed(0)
N = 9
x = np.linspace(0, 6, N) + np.random.randn(N)
x = np.sort(x)
y = x**2 - 4*x - 3 + np.random.randn(N)
x.shape = -1, 1
y.shape = -1, 1

model_1 = Pipeline([
    ('poly', PolynomialFeatures()),
    ('linear', LinearRegression(fit_intercept=False))])
model_2 = Pipeline([
    ('poly', PolynomialFeatures()),
    ('linear', RidgeCV(alphas=np.logspace(-3, 2, 100), fit_intercept=False))])
models = model_1, model_2

#facecolor='w'背景颜色
plt.figure(figsize=(9, 11), facecolor='w')
d_pool = np.arange(1, N, 1)  # 阶
m = d_pool.size
# 获取颜色
clrs = []  
for c in np.linspace(16711680, 255, m):
    clrs.append('#%06x' % int(c))
#线段大小
line_width = np.linspace(5, 2, m)
titles = '线性回归', 'Ridge回归'
for t in range(2):
    model = models[t]
    plt.subplot(2, 1, t+1)    
    #zorder=N控制绘图顺序
    plt.plot(x, y, 'ro', ms=10, zorder=N)
    for i, d in enumerate(d_pool):
            model.set_params(poly__degree=d)
            model.fit(x, y)
            lin = model.get_params('linear')['linear']
            if t == 0:
                print ('%d阶，系数为：' % d, lin.coef_.ravel())
            else:
                print ('%d阶，alpha=%.6f，系数为：' % (d, lin.alpha_), lin.coef_.ravel())
            x_hat = np.linspace(x.min(), x.max(), num=100)
            x_hat.shape = -1, 1
            y_hat = model.predict(x_hat)
            s = model.score(x, y)
            #print(s)
            zorder = N - 1 if (d == 2) else 0
            plt.plot(x_hat, y_hat, color=clrs[i], lw=line_width[i], label=(u'%d阶，score=%.3f' % (d, s)), zorder=zorder)
    plt.legend(loc='upper left')
    plt.grid()
    plt.title(titles[t], fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    plt.suptitle(u'多项式曲线拟合', fontsize=18)
    plt.show()