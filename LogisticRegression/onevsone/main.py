import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from statistics import mode
from sklearn.multiclass import OneVsOneClassifier
import itertools as it
from onevsone import *

# one vs one
iris = datasets.load_iris()
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

y = df['target']
y = np.array(y[:,np.newaxis]).astype(int)
X = np.c_[iris['data']]

split = int(0.8 * 50)
x_train = np.concatenate((X[ : split] , X[50 : 50+split], X[100 : 100 + split]))
x_test  =  np.concatenate((X[split : 50] , X[50+split : 100],X[100 + split :]))
y_train = np.concatenate((y[ : split] , y[50 : 50+split],y[100 : 100+split]))
y_test = np.concatenate((y[split : 50] , y[50+split :100 ],y[100+split :]))

params , cost = train(x_train,y_train)

predition1 = predict(x_test,params)
score1 = score(x_test,y_test,params)
print()
print("the accuracy of the model in test phase is {:.2f}".format(score1))

predition2 = predict(x_train,params)
score2 = score(x_train,y_train,params)
print()
print("the accuracy of the model in trian phase is {:.2f}".format(score2))
