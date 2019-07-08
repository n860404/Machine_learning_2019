# -*- coding: utf-8 -*-

'''Logistic regression'''

import numpy as np
import hw3_1a
import matplotlib.pyplot as plt

def generate_data(n, mean_x, var_x, mean_y, var_y):

    x = hw3_1a.normal_generating(mean_x, var_x, n)
    y = hw3_1a.normal_generating(mean_y, var_y, n)

    return np.vstack((x, y)).T


def plot(n_plot, title, position, x, y):

    ax=plt.subplot(1, n_plot, position)
    ax.set_title(title)
    ax.plot(x[y == 0][:, 0], x[y == 0][:, 1], 'ro')
    ax.plot(x[y == 1][:, 0], x[y == 1][:, 1], 'bo')
    return ax

def preprocess(d):
    l=d.shape[0]
    return np.hstack((d, np.ones((l, 1))))

def Gradient_descent(d,l):
    weights=np.random.rand(d.shape[1])

    while True:
        pre_weights=weights
        l_rate=0.01
        logistic = 1 / (1 + np.exp(-d.dot(weights)))
        gradient = (d.T).dot(l_rate*(l - logistic))
        weights = weights + gradient

        #converge
        if (np.absolute(weights-pre_weights) < 0.0001).all():
            break
    #predict
    l = 1 / (1 + np.exp(-np.dot(d, weights)))
    l=np.around(l)
    return l,weights

def calc_confusion_mat(target, prediction):

    tp = ((target == 1) & (prediction == 1)).sum()
    tn = ((target == 0) & (prediction == 0)).sum()
    fp = ((target == 0) & (prediction == 1)).sum()
    fn = ((target == 1) & (prediction == 0)).sum()
    return tp, tn, fp, fn

def output(title,weights,real,predict):
    print(title)
    print('w:',*weights,'\n')
    tp,tn,fp,fn=calc_confusion_mat(real, predict)
    print('Confusion Matrix:')
    print('\t\tPredict cluster 1\tPredict cluster 2')
    print('Is cluster 1\t\t',tp,'\t\t\t',fp)
    print('Is cluster 2\t\t',fn,'\t\t\t',tn,'\n')
    print('Sensitivity (Successfully predict cluster 1):',tp / (tp + fn))
    print('Specificity (Successfully predict cluster 2):',tn / (tn + fp))

def Newton_method(d,l):
    A=d.copy()
    weights=np.random.rand(d.shape[1])

    while True:
        pre_weights=weights
        z=d.dot(weights)
        v=np.exp(-z) / ((1 + np.exp(-z))**2)
        D=np.diag(v)
        H=(A.T).dot(D).dot(A)
        l_rate=0.01
        logistic = 1 / (1 + np.exp(-d.dot(weights)))
        gradient = (d.T).dot(l_rate*(l - logistic))
        weights = weights + np.linalg.inv(H).dot(gradient)

        if (np.absolute(weights-pre_weights) < 0.01).all():
            break
    l = 1 / (1 + np.exp(-np.dot(d, weights)))
    l=np.around(l)
    return l,weights

def main():
    N=int(input("N="))
    mean_var_pairs1=[float(x) for x in input("mx1,vx1,my1,vy1=").split()]
    mean_var_pairs2=[float(x) for x in input("mx2,vx2,my2,vy2=").split()]

    # Generate data points
    d1 = generate_data(N, *mean_var_pairs1)
    d2 = generate_data(N, *mean_var_pairs2)
    l1 = np.zeros(N)
    l2 = np.ones(N)
    d = np.concatenate((d1, d2))
    l = np.concatenate((l1, l2))

    plot(3, "Ground truth", 1, d, l)

    #Gradient descent
    d=preprocess(d)
    predict1,weights1=Gradient_descent(d,l)
    output("Gradient descent",weights1,l,predict1)
    plot(3, "Gradient descent", 2, d, predict1)

    #Newton's method
    Newton_method(d,l)
    predict2,weights2=Gradient_descent(d,l)
    output("Newton's method",weights2,l,predict2)
    plot(3, "Newton's method", 3, d, predict2)



if __name__ == '__main__':
    main()

#https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8-%E7%B5%B1%E8%A8%88%E5%AD%B8%E7%BF%92-%E7%BE%85%E5%90%89%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-aff7a830fb5d
#http://cpmarkchang.logdown.com/posts/189108-machine-learning-perceptron-algorithm