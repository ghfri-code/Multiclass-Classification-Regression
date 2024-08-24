import numpy as np
from statistics import mode

alpha = 0.001
precision = 0.00001

def train( X, y):

        alltheta = []
        allcost = []
        for k1 in (np.unique(y)):
            for k2 in np.arange(k1+1,len(np.unique(y))):
                data_k = data_one_vs_one(k1, k2, X, y)
                y_k = data_k[0]
                X_k = data_k[1]
                theta , cost = fit(X_k, y_k)
            alltheta.append((theta, k1))
            allcost.append((cost,k1))

        return alltheta , allcost

#This function splits the dataset for binary classification

def data_one_vs_one(k1, k2, X_train, y_train):
        indexes_k1 = (y_train == k1)
        indexes_k2 = (y_train == k2)
        y_train_k = np.concatenate((y_train[indexes_k1], y_train[indexes_k2]))
        indexes__k1 = (np.where(y_train == k1))
        indexes__k2 = (np.where(y_train == k2))
        X_train_1 = X_train[indexes__k1,:]
        X_train_1 = X_train_1[0,:,:]
        X_train_2 = X_train[indexes__k2,:]
        X_train_2 = X_train_2[0,:,:]
        X_train_k = np.concatenate((X_train_1, X_train_2))
        y_train_k = one_vs_one_transformed_labels(k1,k2,y_train_k)
        y_train_k = y_train_k[:,np.newaxis]

        return y_train_k, X_train_k


#This function changes the labels of selected classes for training into 0 and 1

def one_vs_one_transformed_labels(k1, k2, y_train_k):
    y = np.zeros(y_train_k.shape[0])
    for i in np.arange(y_train_k.shape[0]):
        if y_train_k[i] == k1:
            y[i] = 0
        else:
            y[i] = 1
    return y

#This function is resonsible for calculating the sigmoid value with given parameter

def sigmoid_function(x):
    value = 1 / (1 + np.exp(-x))
    return value


 # The fuctions calculates the cost value

def _cost_function(h,theta, y):

    return -(y*np.log(h) + (1-y)*np.log(1-h)).mean()


# This function calculates the theta value by gradient ascent

def _gradient_ascent(X,h,theta,y,m):
    gradient_value = np.dot(X.T, (y - h)) / m
    theta = theta +  alpha * gradient_value
    return theta


#This function calculates the optimal theta value using which we predict the future data

def fit(X, y):
    print("Fitting the given dataset..")
    X = np.insert(X, 0, 1, axis=1)
    m = len(y)
    theta = np.zeros((X.shape[1],1))
    cost = []
    for count in range(100000):
        z = np.dot(X,theta)
        h = sigmoid_function(z)
        theta = _gradient_ascent(X,h,theta,y,m)
        cost.append(_cost_function(h,theta,y))
        if count != 0:
            diff = abs(cost[count] - cost[count - 1])

            if diff < precision:
                print("Converged iteration: ",count)
                break

    return theta, cost


# this function predict the label for new data based on majority count of classifiers

def predict(X,theta_):
    X = np.insert(X, 0, 1, axis=1)
    label = 0
    X_predicted=[]
    for i in X:
        l1 = []
        for th , c in theta_:
            if (c == 0):
                z = np.dot(i,th)
                sig = (sigmoid_function(z),c)
                if sig[0] < .5:
                    label = 0
                else:
                    label = 1
                l1.append(label)

            if (c == 1):
                z = np.dot(i,th)
                sig = (sigmoid_function(z),c)
                if sig[0] < .5:
                    label = 0
                else:
                    label = 2
                l1.append(label)

            if (c == 2):
                z = np.dot(i,th)
                sig = (sigmoid_function(z),c)
                if sig[0] < .5:
                    label = 1
                else:
                    label = 2
                l1.append(label)


        X_predicted.append(mode(l1))

    X_predicted =np.array(X_predicted)
    X_predicted = X_predicted[:,np.newaxis]
    return X_predicted


#This function compares the predictd label with the actual label to find the model performance

def score(X, y,param_):
    score = np.sum(predict(X,param_) == y) / len(y)
    return score
