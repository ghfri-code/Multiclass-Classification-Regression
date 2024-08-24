import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

precision = 0.000001

class LogisticRegression(object):

    def __init__(Logreg, alpha=0.001, n_iteration=100):  #This function intializes the alpha value and iteration value
        Logreg.alpha = alpha
        Logreg.n_iter = n_iteration

    def sigmoid_function(Logreg, x): #This function calculates the sigmoid value with given parameters
        value = 1 / (1 + np.exp(-x))
        return value

    def _cost_function(Logreg,h,theta, y): # The fuctions calculates the cost value

        return -(y*np.log(h) + (1-y)*np.log(1-h)).mean()


    def _gradient_ascent(Logreg,X,h,theta,y,m): # This function calculates the theta value by gradient ascent
        gradient_value = np.dot(X.T, (y - h)) / m
        theta += Logreg.alpha * gradient_value
        return theta


    def fit(Logreg, X, y): #This function calculates the optimal theta value using which we predict the future data
        print("Fitting the given dataset..")
        Logreg.theta = []
        Logreg.cost = []
        X = np.insert(X, 0, 1, axis=1)
        m = len(y)
        for i in np.unique(y):
            print('Acscending the gradient for label type ' + str(i) + ' vs Rest')
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros((X.shape[1],1))
            cost = []
            for count in range(Logreg.n_iter):
                z = np.dot(X,theta)
                h = Logreg.sigmoid_function(z)
                theta = Logreg._gradient_ascent(X,h,theta,y_onevsall,m)
                cost.append(Logreg._cost_function(h,theta,y_onevsall))
                if count != 0:
                    diff = abs(cost[count] - cost[count - 1])
                    if diff < precision:
                        print("Converged iteration: ",count,i)
                        break

            Logreg.theta.append((theta, i))
            Logreg.cost.append((cost,i))

        return Logreg

    def predict(Logreg, X): # this function calls the max predict function to classify the given test data

        X = np.insert(X, 0, 1, axis=1)
        X_predicted = [max((Logreg.sigmoid_function(i.dot(theta)), c) for theta, c in Logreg.theta) for i in X ]

        return np.array(X_predicted)

    def score(Logreg,X, y): #This function compares the predictd label with the actual label to find the model performance
        score = np.sum(Logreg.predict(X) == y) / len(y)
        return score

    def _plot_cost(Logreg,costh): # This function plot the Cost function value
        for cost,c in costh   :
            plt.plot(range(len(cost)),cost,'r')
            plt.title("Convergence Graph of Cost Function of type " + str(c) +" vs All")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Cost")
            plt.show()
