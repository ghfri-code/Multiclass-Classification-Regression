# Regression (multi classification)
**Dataset** : Iris (https://archive.ics.uci.edu/ml/datasets/Iris)

In this project, we use the iris dataset for multiclass classification(one-vs.-one and one-vs.-all) by logistic regression. Then use softmax regression and compare them. Consider the first 80% of the data in each class for the train and the rest 20% for the test.

## comparing methods
### What method (one-vs.-one, one-vs.-all, or softmax) has worked best? ###
- Among the three models, the softmax regression method performed better than the other two and had the highest accuracy in the training phase.
- In the onevsone method, we have more subsets of data to train our models than the onevsall method. For example, if the data is classified into N classes, in the onevsone method, we need N*(N-1)/2 subsets of the data, and in the onevsall method, we need N subsets, and also in the onevsone algorithm, the number of iterations is less than that of onevsall has converged. 
- Another issue that is raised in the onevsall method is the issue of data imbalance. The state where the number of samples in one class is much less than the samples of another class is called an imbalanced state, and this imbalance causes the quality of the binary classification algorithm to decrease. Therefore, the onevsall algorithm cannot provide the desired accuracy and quality.
