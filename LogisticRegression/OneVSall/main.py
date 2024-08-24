import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from Onevsall import *

# one vs all
iris = datasets.load_iris()
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

y = df['target']
y = np.array(y[:,np.newaxis])
X = np.c_[iris['data']]
y_unique = np.unique(y)


split = int(0.8 * 50)
x_train = np.concatenate((X[ : split] , X[50 : 50+split], X[100 : 100 + split]))
x_test  =  np.concatenate((X[split : 50] , X[50+split : 100],X[100 + split :]))
y_train = np.concatenate((y[ : split] , y[50 : 50+split],y[100 : 100+split]))
y_test = np.concatenate((y[split : 50] , y[50+split :100 ],y[100+split :]))

logi = LogisticRegression(n_iteration=100000).fit(x_train, y_train)

predition1 = logi.predict(x_train)
score1 = logi.score(x_train,y_train)
print()
print("the accuracy of the model in train pahase is {:.2f}".format(score1*100))


predition2 = logi.predict(x_test)
score2 = logi.score(x_test,y_test)
print()
print("the accuracy of the model in test phase is {:.2f}".format(score2*100))

#Plot of cost function
logi._plot_cost(logi.cost)
