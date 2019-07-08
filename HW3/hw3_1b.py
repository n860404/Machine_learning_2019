# -*- coding: utf-8 -*-
'''Polynomial basis linear model data generator'''

import numpy as np
import hw3_1a

def polynomial(basis,var,weights,n=1):

    noise = hw3_1a.normal_generating(0, var)
    x = np.random.uniform(-1, 1, n)

    X=[]
    for power in range(basis):
        X.append( x[:] ** power)

    X=np.array(X)
    weights=np.array(weights)
    y=weights.dot(X)+ noise


    return y



def main():
     basis=int(input("n=")) #basis num
     var=float(input("a=")) #var
     weights=[float(x) for x in input("w=").split()] #weight list
     #n=1 #Number of data to generate
     values = polynomial(basis,var,weights)

     print(*values, sep=' ')


if __name__ == '__main__':
    main()