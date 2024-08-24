import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MultiClassLogisticRegression:

    def __init__(self, n_iter = 100000, thres=0.00001):  #This function intializes the threshold value for convergene and iteration value
        self.n_iter = n_iter
        self.thres = thres


    def fit(self, X, y, lr=0.00001):
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        y = self.one_hot(y)
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
        self.fit_data(X, y, lr)


    def fit_data(self, X, y, lr):
        i = 0
        while (i < self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict(X)))
            error = y - self.predict(X)
            update = (lr * np.dot(error.T, X))
            self.weights += update
            diff = np.abs(self.loss[i] - self.loss[i - 1])
            if (diff < self.thres) and (i != 0):
                print('Training Accuray at {} iterations is {:.2f}'.format(i, self.accuracy(X, y)))
                break
            i +=1


    def predict(self, X):    #This function calculates the probabilities with given parameters
        pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
        return self.softmax(pre_vals)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

    def predict_classes(self, X):   # this function returns the index on which the probability is maximum
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))



    def one_hot(self, y):  #this function encode the y vector to be compatible with multiclass problem
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]


    def accuracy(self, X, y):   #This function compares the predictd label with the actual label to find the model performance
        return np.mean(np.argmax(self.predict(X), axis=1) == np.argmax(y, axis=1))

    def cross_entropy(self, y, probs):      # The fuctions calculates the cost value
        return -1 * np.mean(y * np.log(probs))


