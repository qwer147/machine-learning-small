# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 20:24:38 2018

@author: Administrator
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

iris = load_iris()
x,y = iris.data,iris.target

#只使用前两列
x = x[:,:2]

#注意顺序
train_data,test_data,train_labels,test_labels = train_test_split(x,y,test_size=0.3, random_state=1)
model = Pipeline([('sc',StandardScaler()),
                  ('dtc',DecisionTreeClassifier(criterion='entropy',max_depth=3))])
model = model.fit(train_data,train_labels)

y_pred = model.predict(test_data)

#保存模型
f = open('D:/git使用/回归1/iris_tree.dot','w')
w = tree.export_graphviz(model.get_params('dtc')['dtc'],out_file=None,
                         feature_names=iris.feature_names[:2],class_names=iris.target_names,filled=True, rounded=True,special_characters=True)
f.close()
'''
import pydotplus
from sklearn import tree
#from IPython.display import Image
graph = pydotplus.graph_from_dot_data(w)
#img = Image(graph.create_png())
graph.write_png("out.png")
'''

#画图
N, M = 100, 100  
x1_min, x1_max = x[:, 0].min(), x[:, 0].max() 
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)  
x_show = np.stack((x1.flat, x2.flat), axis=1)  

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
y_show_hat = model.predict(x_show)  
y_show_hat = y_show_hat.reshape(x1.shape)
  
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
plt.scatter(test_data[:, 0], test_data[:, 1], c=y_pred.ravel(), edgecolors='k', s=100, cmap=cm_dark, marker='o')  # 测试数据
plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
plt.xlabel(iris.feature_names[0], fontsize=15)
plt.ylabel(iris.feature_names[1], fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(True)
plt.title('鸢尾花数据的决策树分类', fontsize=17)
plt.show()


result = (y_pred == test_labels)
acc = np.mean(result)

#找最好的深度
depth = np.arange(1,15)
err_list = []
for d in depth:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
    clf.fit(train_data,train_labels)
    y_pred = clf.predict(test_data)
    acc = np.mean(y_pred==test_labels)
    err = 1-acc
    err_list.append(err)
    print('在树深为%d时,准确率是%.2f%%'%(d,acc*100))
plt.figure(facecolor='w')
plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel('决策树深度', fontsize=15)
plt.ylabel('错误率', fontsize=15)
plt.title('决策树深度与错误率', fontsize=17)
plt.grid(True)
plt.show()

#随机森林
x,y,feature = iris.data,iris.target,iris.feature_names

#
clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=4)
rf_clf = clf.fit(x, y.ravel())

y_pred = rf_clf.predict(x)
print(np.sum(y_pred!=y))