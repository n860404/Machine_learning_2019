# -*- coding: utf-8 -*-
"""Baysian Linear regression"""

import numpy as np
import hw3_1a
import matplotlib.pyplot as plt
def polynomial(basis,var,weights,n=1):

    noise = hw3_1a.normal_generating(0, var)
    x = np.random.uniform(-1, 1, n)
    X=[]
    for power in range(basis):
        X.append( x[:] ** power)

    X=np.array(X)
    weights=np.array(weights)
    y=weights.dot(X)+ noise

    return x,y,X

def graph_polynomial(weights,noise):

    x=np.arange(-2,2,0.1)
    n=len(weights)
    X=[]
    for power in range(n):
        X.append( x[:] ** power)
    weights=np.array(weights)
    y=weights.dot(X)+noise

    return x,y
def graph_polynomial_var(weights,var_ma,a):
    x=np.arange(-2,2,0.1)
    n=len(weights)
    X=[]
    v1=[]
    v2=[]
    weights=np.array(weights)
    for power in range(n):
        X.append( x[:] ** power)
    Y=np.array(X)
    Y=Y.T
    for i in range(40):
        X=Y[i]

        var=1/a + ((X).dot(var_ma)).dot(X.T)
        y1=weights.dot(X)+var
        v1.append(y1)
        y2=weights.dot(X)-var
        v2.append(y2)
    #print(v1)
    '''
    print(X)
    var=1/a + ((X.T).dot(var_ma)).dot(X)
    y1=weights.dot(X)+var
    y2=weights.dot(X)-var
    #print(var)
    '''
    return x,v1,v2


def is_converge(prior_cov,cov_ma):
    ch=1
    error=0.00001
    ch_cov=prior_cov-cov_ma

    for i in ch_cov:
        for j in i:
            if(abs(j)>error):
                ch=0
                break
    return ch


def main():
    b=int(input("b="))
    basis=int(input("n=")) #basis num
    v=float(input("a=")) #var
    weights=[float(x) for x in input("w=").split()] #weight list
    I=np.eye(basis)
    precision=(1/b)*I
    a=1/v
    time=0
    check=0
    x_real=[]
    y_real=[]
    x_10=[]
    y_10=[]
    x_50=[]
    y_50=[]
    mean_fi=[]
    var_fi=[]
    mean_10=[]
    var_10=[]
    mean_50=[]
    var_50=[]
    prior_mean=np.zeros((basis,1))
    prior_cov=precision
    while True:
        time+=1
        '''
        x=-0.64152
        y=0.19039
        X=np.array([[1],[x],[x**2],[x**3]])
        '''
        x,y,X= polynomial(basis,v,weights)
        print("Add data point ("+str(float(x))+','+str(float(y))+')\n')

        if time==1:
            B=a*X.dot((X.T))+b*I
            cov_ma=np.linalg.inv(B)
            mean_ma=a*(cov_ma).dot(X)*float(y)
        elif time>1:
            '''
            if time==2:
                x=0.07122
                y=1.63175
                X=np.array([[1],[x],[x**2],[x**3]])
            elif time==3:
                x=-0.19330
                y=0.24507
                X=np.array([[1],[x],[x**2],[x**3]])
            else:
                break
            '''
            prior_mean=mean_ma
            prior_cov=cov_ma
            m=mean_ma
            S=np.linalg.inv(cov_ma)
            B=a*X.dot((X.T)) + B.dot(I)
            cov_ma=np.linalg.inv(B)
            mean_ma=cov_ma.dot(a*X*float(y)+S.dot(m))
            check=is_converge(prior_cov,cov_ma)

        #print("Add data point ("+str(float(x))+','+str(float(y))+')\n')
        mean=(X.T).dot(prior_mean)
        var=1/a + ((X.T).dot(prior_cov)).dot(X)
        print("Postirior mean:")
        for i in range(basis):
            print(*mean_ma[i], sep='\n')
        print("\nPostirior variance:")
        for i in range(basis):
            print(*cov_ma[i], sep=' , ')

        print("\nPredictive distribution ~ N("+str(float(*mean))+','+str(float(*var))+')')
        print("--------------------------------------------------")

        x_real.append(float(x))
        y_real.append(float(y))
        if time<=10:
            x_10.append(float(x))
            y_10.append(float(y))
            if time==10:
                mean_10=mean_ma.T
                var_10=prior_cov
        if time<=50:
            x_50.append(float(x))
            y_50.append(float(y))
            if time==50:
                mean_50=mean_ma.T
                var_50=prior_cov

        #判斷probability converged
        if(check==1):
            mean_fi=mean_ma.T
            var_fi=prior_cov
            break

    #Ground truth

    x,y=graph_polynomial(weights,0)
    x_1,y_1=graph_polynomial(weights,v)
    x_2,y_2=graph_polynomial(weights,-v)
    #plt.figure()
    plt.subplot(2,2,1)
    #plt.plot([0,1],[0,1])
    plt.plot(x,y, 'k')
    plt.plot(x_1,y_1, 'r')
    plt.plot(x_2,y_2, 'r')
    plt.xlim(-2, 2)
    plt.ylim(-20, 30)
    plt.title("Ground truth")


    #predict result

    x,y=graph_polynomial(*mean_fi,0)
    x_1,y_1,y_2=graph_polynomial_var(*mean_fi,var_fi,a)
    plt.subplot(2,2,2)

    plt.scatter(x_real, y_real, alpha=0.5)
    plt.plot(x,y, 'k')
    plt.plot(x_1,y_1, 'r')
    plt.plot(x_1,y_2, 'r')
    plt.xlim(-2, 2)
    plt.ylim(-20, 30)
    plt.title("Predict Result")



    #After 10 incomes
    x,y=graph_polynomial(*mean_10,0)
    x_1,y_1,y_2=graph_polynomial_var(*mean_10,var_10,a)
    plt.subplot(223)

    plt.scatter(x_10, y_10, alpha=0.5)
    plt.plot(x,y, 'k')
    plt.plot(x_1,y_1, 'r')
    plt.plot(x_1,y_2, 'r')
    plt.xlim(-2, 2)
    plt.ylim(-20, 30)
    plt.title("After 10 incomes")


    #After 50 incomes
    x,y=graph_polynomial(*mean_50,0)
    x_1,y_1,y_2=graph_polynomial_var(*mean_50,var_50,a)
    plt.subplot(224)


    plt.plot(x,y, 'k')
    plt.plot(x_1,y_1, 'r')
    plt.plot(x_1,y_2, 'r')
    plt.scatter(x_50, y_50, alpha=0.5)
    plt.xlim(-2, 2)
    plt.ylim(-20, 30)
    plt.title("After 50 incomes")
    plt.show()


if __name__ == '__main__':
    main()

#https://github.com/zjost/bayesian-linear-regression/blob/master/src/bayes-regression.ipynb