# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:50:35 2018

@author: Administrator
"""
#网格搜索
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
#加载模型
from sklearn.linear_model import Lasso,Ridge

#读取数据
f = open('D:/alldata/回归案例数据/Advertising.csv')
data = pd.read_csv(f,index_col=0)

#划分x,y
x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

model_lasso = Lasso()
#model_ridge  = Ridge()

#参数
#等比数列，前面两个表示开始位置和停止位置但都是10**x，最后一个数字表示取几个数字
alpha_can = np.logspace(-3,2,10)


#寻找最优参数
model = GridSearchCV(model_lasso,param_grid={'alpha':alpha_can},cv=5)
model.fit(x_train,y_train)
print(model.best_params_)
print(model.best_score_)

#直接可以用GridSearchCV去做预测已经选好了最好参数
pred = model.predict(x_test)
mse = mean_squared_error(pred,y_test)
print('测试集上的误差%f'%mse)
#画图
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, pred, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()

#学习率曲线
from sklearn.model_selection import learning_curve
#from sklearn.model_selection import ShuffleSplit

'''
   estimator分类器
'''
def plot_learning_curve(estimator,title,X,y,ylim=None,
                        cv=None,n_jobs=1,train_sizes=np.linspace(0.1,1,5)):
    plt.figure()
    #返回训练集大小，得分测试集得分
    train_sizes,train_scores,test_scores=learning_curve(
            estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    
    #计算方差，均值为画图做准备
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    
    #填充
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,alpha=.1,
                     color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,alpha=.1,
                     color='g')
    #画图
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',
             label='Training score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',
             label='Cross-validation score')
    
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()
plot_learning_curve(model,'learining',x,y,
                        cv=5,n_jobs=-1)
    
'''
   当训练集和测试集的误差收敛但却很高时，为高偏差（注意看图的
   y轴代表的是什么含义）可能欠拟合。
   此时可以增加模型参数，构造更多特征等此时增加数据量
   是没用的
   
   当训练集和测试集误差有较大差距时，为高方差
   当训练集的准确率比其他独立数据集上的测试结果的准确率要高时
   一般都是过拟合
   此时可以增大数据集，降低模型复杂度，增大正则项或通过特征选
   择减少特征数
   
   理想情况是找到偏差和方差都很小的情况，就是收敛且误差较小
   
'''    