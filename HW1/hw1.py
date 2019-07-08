# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def build_A(n,x):
    A_list=[]
    for i in x:
        raw=[]
        for j in range(n):
            t=i**(n-j-1)
            raw.append(t)
        A_list.append(raw)

    A=np.array(A_list)
    return(A)

def LU(A):
    A=list(A)
    n=len(A)
    for j in range(n):
        for k in range(j):
            for i in range(k+1,j):
                A[i][j]=A[i][j]-A[i][k]*A[k][j]

        for k in range(j):
            for i in range(j,n):
                A[i][j]=A[i][j]-A[i][k]*A[k][j]
        for k in range(j+1,n):
            A[k][j]=A[k][j]/A[j][j]

    L=np.eye(n, dtype = 'float')
    U=np.eye(n, dtype = 'float')
    L=list(L)
    U=list(U)
    for i in range(n):
        for j in range(n):
            if i>j:
                L[i][j]=A[i][j]
            elif i<=j:
                U[i][j]=A[i][j]
    #print("L:",L)
    #print("U:",U)
    return L,U

def inverse(A):
    #A=[[-19,20,-6],[-12,13,-3],[30,-30,12]]
    n=len(A)
    I=list(np.eye(n, dtype = 'float'))#I單位矩陣
    X=list(np.zeros((n, n)))
    L,U=LU(A)

    #LY=I
    Y = list( np.zeros((n, n)))
    for j in range(n):
        Y[0][j]=I[0][j]
        for i in range(1,n):
            t=0.0
            for k in range(i):
                t+=L[i][k]*Y[k][j]
            Y[i][j]=I[i][j]-t

    #Ux=Y
    for j in range(n):
        X[n-1][j]=Y[n-1][j]/U[n-1][n-1]
        for i in range(n-2,-1,-1):
            t=0.0
            for k in range(i,n):
                t+=U[i][k]*X[k][j]
            X[i][j]=(Y[i][j]-t)/U[i][i]
    X=np.array(X)
    #print("inv:",X)
    return X

def output(ans,n,y,A):
    num=len(y)
    print("Fitting line: ",end= ' ')
    for i in range(n):
        if i==n-1:
            print(str(ans[i]))
        else:
            print(str(ans[i])+"X^"+str(n-i-1),end= ' ')
            if ans[i+1]>0:
                print("+",end= ' ')
    y_pre=A.dot(ans)
    error=0
    for i in range(num):
        error+=(y[i]-y_pre[i])**2
    print("Total error:"+str(error)+'\n')

def output_graph(A,ans,x,y,title):
    plt.figure()
    y_pre=A.dot(ans)
    plt.plot(x, y,'ro')
    plt.plot(x, y_pre)
    plt.title(title)
    plt.show()

def main():
    #read file
    x=[]
    y=[]
    file_name=input("File name:")
    with open(str(file_name)) as infile:
    #with open('testfile.txt') as infile:
        lines = infile.readlines()
        for line in lines:
            #print(line)
            x.append(float(line.split(',')[0]))
            y.append(float(line.split(',')[1]))

    num=len(x)
    #case=1
    #print ("Case"+str(case)+':',end= ' ' )
    n=int(input("n="))
    lamb=int(input("λ="))

    #LSE
    A=build_A(n,x)
    I=np.eye(n, dtype = 'int')#單位矩陣
    y=np.array(y).T

    inv=inverse( (A.T).dot(A)+lamb*I )

    ans=inv.dot(A.T).dot(y)
    #ans_fun=np.linalg.inv((A.T).dot(A)+lamb*I).dot(A.T).dot(y)#inverse要自己寫
    #print(inv)
    #print(ans)
    #print(np.linalg.inv((A.T).dot(A)+lamb*I))
    #print(ans_fun)

    #print LSE
    print("LSE:")
    output(ans,n,y,A)


    #Newton’s Method
    inv_new=inverse( (A.T).dot(A))
    ans_new=inv_new.dot(A.T).dot(y)
    print("Newton’s Method:")
    output(ans_new,n,y,A)

    #visualization
    #LSE
    output_graph(A,ans,x,y,"LSE")
    #Newton’s Method
    output_graph(A,ans_new,x,y,"Newton’s Method")
if __name__ == "__main__":
    main()