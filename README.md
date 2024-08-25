**Dataset** : Iris https://archive.ics.uci.edu/ml/datasets/Iris

In this part we use iris dataset for multiclass classification(one-vs.-one and one-vs.-all) by logistic regression. Then using softmax regression and compare them. Consider the first 80% of the data in each class for train and the rest 20% for test.

### Plot cost function for enough iteration for one-vs-all method:
![type0](https://github.com/Ghafarian-code/Multiclass-Classification-Regression/blob/main/LogisticRegression/OneVSall/images/Figure_1.png)
![type1](https://github.com/Ghafarian-code/Multiclass-Classification-Regression/blob/main/LogisticRegression/OneVSall/images/Figure_2.png)
![type2](https://github.com/Ghafarian-code/Multiclass-Classification-Regression/blob/main/LogisticRegression/OneVSall/images/Figure_3.png)


### comparing methods
**What method (one-vs.-one, one-vs.-all or softmax) has worked best?**

Among the three models, the soft max regression method has performed better than the other two models with the highest accuracy in the training phase.
In the onevsone method, we have more subsets of data to train our models than the onevsall method. For example, if the data is classified into N classes, in the onevsone method, we need N*(N-1)/2 subsets of the data, and in the onevsall method, we need N subsets, and also in the onevsone algorithm, the number of iterations is less than that of onevsall. has converged. Another issue that is raised in onevsall method is the issue of data imbalance. For example, if we have 10,000 samples that are divided into 20 classes, we need 20 data sets, one by one, each sample of a class. If we compare the samples of other classes, 500 samples from one class will be compared to 9500 samples from other classes. The state where the number of samples in one class is much less than the samples of another class is called unbalanced state, and this imbalance causes the quality of the binary classification algorithm to decrease and finally the final algorithm, which is the onevsall algorithm, cannot Provide the desired accuracy and quality.
