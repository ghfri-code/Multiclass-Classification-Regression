import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from softmaxregression import *


iris = datasets.load_iris()
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

y = df['target']
y = np.array(y[:,np.newaxis])
X = np.c_[iris['data']]
bias = np.ones((X.shape[0], 1))
X = np.concatenate((bias, X), axis=1)
y_unique = np.unique(y)
n = X.shape[1]
precision = 0.000001

split = int(0.8 * 50)
x_train = np.concatenate((X[ : split] , X[50 : 50+split], X[100 : 100 + split]))
x_test  =  np.concatenate((X[split : 50] , X[50+split : 100],X[100 + split :]))
y_train = np.concatenate((y[ : split] , y[50 : 50+split],y[100 : 100+split]))
y_test = np.concatenate((y[split : 50] , y[50+split :100 ],y[100+split :]))
K = np.unique(y)
m , n = X.shape

lr = MultiClassLogisticRegression(thres=0.0000001)
lr.fit(x_train,y_train,lr=0.0001)

y_test1 = lr.one_hot(y_test)
print()
print("Test Accuray is {:.2f}".format(lr.accuracy(x_test,y_test1)))

