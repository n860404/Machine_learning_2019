# -*- coding: utf-8 -*-
'''Univariate gaussian data generator'''

import numpy as np

def std_normal(n):

    u=np.random.uniform(size=n)
    v=np.random.uniform(size=n)

    z0=((-2)*np.log(u))**0.5 * np.cos(2*np.pi*v)

    return z0

def normal_generating(mean,var,n=1):

    std_dev=var**0.5 #標準差
    value=mean + std_dev * std_normal(n) # X = μ + σZ
    return value

def main():

    mean=float(input("mean:"))
    var=float(input("variance:"))
    #n=1 #Number of data to generate

    values = normal_generating(mean,var)
    print(*values, sep=' ')

if __name__ == '__main__':
    main()

#https://en.wikipedia.org/wiki/Normal_distribution
#https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
#https://medium.com/@balamurali_m/normal-distribution-with-python-793c7b425ef0
#https://www.alanzucconi.com/2015/09/16/how-to-sample-from-a-gaussian-distribution/