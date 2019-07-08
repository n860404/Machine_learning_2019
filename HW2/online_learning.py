# -*- coding: utf-8 -*-
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return (n*factorial(n-1))
def compute_likelihood(data):
    data_len=len(data)
    zero=0
    for i in data:
        if i=='0':
            zero+=1
    one=data_len-zero
    p0=zero/data_len
    p1=one/data_len
    likelihood=(factorial(data_len)/factorial(zero)/factorial(one))*(p0**zero)*(p1**one)

    return likelihood ,one,zero
def main():
    file_name="testfile.txt"
    data=[]
    with open(str(file_name)) as infile:
        lines = infile.readlines()
        for line in lines:
            line = line . strip ( '\n' )
            data.append(line)
    a=int(input("a="))
    b=int(input("b="))
    prior_a=a
    prior_b=b
    num=len(data)
    for i in range(num):
        print("case "+str(i+1)+": "+data[i])
        likelihood,one,zero=compute_likelihood(data[i])
        print("Likelihood: ",likelihood)
        print("Beta prior: a="+str(prior_a)+" b="+str(prior_b))
        prior_a+=one
        prior_b+=zero
        print("Beta posterior: a="+str(prior_a)+" b="+str(prior_b)+'\n')





if __name__ == "__main__":
    main()