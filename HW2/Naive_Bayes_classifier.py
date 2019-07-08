import struct
import gzip
import numpy as np
import math
import copy
def read_files(train_feature_file,train_label_file,test_feature_file,test_label_file):
    with gzip.open(train_feature_file) as fp:
        header = struct.unpack(">IIII", fp.read(16))
        _, N, rows, cols = header

        imgs = np.fromstring(fp.read(N*rows*cols), dtype=np.uint8)

        imgs = imgs.reshape((N, rows*cols))

    with gzip.open(test_feature_file) as fp:
        header = struct.unpack(">IIII", fp.read(16))
        _, N, rows, cols = header

        test_imgs = np.fromstring(fp.read(N*rows*cols), dtype=np.uint8)

        test_imgs = test_imgs.reshape((N, rows*cols))

    with gzip.open(train_label_file) as fp:
        header = struct.unpack(">II", fp.read(8))
        _, N = header

        labels = np.fromstring(fp.read(N), dtype=np.uint8)

    with gzip.open(test_label_file) as fp:
        header = struct.unpack(">II", fp.read(8))
        _, N = header

        test_labels = np.fromstring(fp.read(N), dtype=np.uint8)

    return imgs, labels, test_imgs, test_labels
def compute_prior(labels):
    prior=[0]*10
    len_label=len(labels)
    for i in labels:
        prior[i]+=1
    for i in range(10):
        prior[i]=prior[i]/len_label

    return prior
def trans_bin(imgs):
    bin_num=8
    bin_imgs=[]
    for i in imgs:
        part=[]
        for j in i:
            part.append(math.ceil((j+1)/8)-1)#換成0-31
        bin_imgs.append(part)
    return bin_imgs
def compute_likelihood(labels,imgs,bin_num):
    likelihood=[]
    len_label=len(labels)
    pixcel_num=28*28
    for i in range(10):#label:0-9
        #part=[[0]*bin_num]*pixcel_num
        part=[ [1] * bin_num for i in range(pixcel_num)]
        num=0
        for j in range(len_label):
            if i == labels[j]:#現在要建的label
                num=num+1
                for k in range(pixcel_num):
                    part[k][imgs[j][k]]+=1

        for p in range(pixcel_num):
            s=sum(part[p])
            for q in range(bin_num):
                part[p][q]=part[p][q]/s
                '''
                if part[p][q]==0:
                    part[p][q]=1/num
                '''
        #print(part)
        likelihood.append(part)

    return likelihood

def compute_posterior(data,label,prior,likelihood):
    data_prior=-(math.log(prior[label]))
    data_likelihood=0
    for i in range(len(data)):
        l=likelihood[label][i][int(data[i])]
        data_likelihood+=-(math.log(l))
    posterior=data_prior+data_likelihood
    return posterior

def predict(test_imgs,test_labels,prior,likelihood):
    #test_imgs=test_imgs[0:10]
    test_len=len(test_imgs)
    error=0
    for i in range(test_len):
        data=test_imgs[i]
        target=test_labels[i]
        posterior=[]
        for j in range(10):
            posterior.append(compute_posterior(data,j,prior,likelihood))
        for j in range(10):
            #posterior=normalize(posterior)
            print(str(j)+':'+str(posterior[j]))
        minpos = posterior.index(min(posterior))
        print("Predicton:"+str(minpos)+" Ans:"+str(target))
        if minpos!=int(test_labels[i]):
            error+=1

    print("error rate:",error/test_len)

def graph(likelihood):
    pixcel_num=28*28
    predict_graph=[]
    for i in range(10):
        part=[]

        for j in range(pixcel_num):
            zero=0.0
            one=0.0
            for k in range(32):
                if k<=15:
                    zero+=likelihood[i][j][k]

                else:
                    one+=likelihood[i][j][k]

            if zero>=one:
                part.append(0)
            else:
                part.append(1)
        predict_graph.append(part)

    for i in range(10):
        print(str(i)+':')
        n=0
        for j in predict_graph[i]:

            n+=1
            print(str(j), end=' ')
            if n==28:
                n=0
                print('\n')

def normalize(data):
    s=sum(data)
    for i in range(10):
        data[i]=data[i]/s
    return data


def compute_mean_deviation(labels,imgs):
    pixcel_num=28*28
    data_len=len(labels)
    mean=[]
    deviation=[]
    s=0
    for i in range(10):#label:0-9
        part=[0]*pixcel_num
        num=0
        for j in range(data_len):
            if i==labels[j]:
                num=num+1
                for k in range(pixcel_num):
                    part[k]+=imgs[j][k]
        for p in range(pixcel_num):
            part[p]=part[p]/num
        mean.append(part)

    for i in range(10):#label:0-9
        part=[0]*pixcel_num
        num=0
        for j in range(data_len):
            if i==labels[j]:
                num=num+1
                for k in range(pixcel_num):
                    part[k]+=imgs[j][k]**2

        for p in range(pixcel_num):
            part[p]=math.sqrt((part[p]/num)-(mean[i][p]**2))
        deviation.append(part)



    return mean,deviation

def compute_Gaussian_likelihood(mean,deviation,data,j):
    likelihood=[]
    nan=0
    for i in range(len(data)):
        m=mean[j][i]#j=label
        d=deviation[j][i]
        if d!=0:
            k=1/(d*(math.sqrt(2*math.pi)))
            l1=math.log(k) + ( (-((data[i]-m)**2)/(2*d*d)) * math.log(math.exp(1)) )
            likelihood.append(l1)
        elif d==0:
            nan+=1

    minimum=min(likelihood)
    data_likelihood=-(sum(likelihood)+minimum*nan)
    return data_likelihood

def continuous_predict(imgs,labels,test_imgs,test_labels,prior):

    mean,deviation=compute_mean_deviation(labels,imgs)
    test_len=len(test_imgs)
    error=0

    for i in range(test_len):
        data=test_imgs[i]
        target=test_labels[i]
        posterior=[]
        for j in range(10):
            data_likelihood=compute_Gaussian_likelihood(mean,deviation,data,j)
            data_prior=-(math.log(prior[j]))
            data_posterior=data_likelihood+data_prior
            posterior.append(data_posterior)
        for j in range(10):
            posterior=normalize(posterior)
            print(str(j)+':'+str(posterior[j]))
        minpos = posterior.index(min(posterior))
        print("Predicton:"+str(minpos)+" Ans:"+str(target))
        if minpos!=target:
            error+=1
    print("error rate:",error/test_len)

    #print graph
    predict_graph=[]
    pixcel_num=28*28

    for i in range(10):
        part=[]
        for j in range(pixcel_num):
            zero=0.0
            one=0.0
            m=mean[i][j]
            d=deviation[i][j]
            find=[]
            p0=0
            p1=0
            for k in range(256):
                if d!=0:
                    p=1/(d*(math.sqrt(2*math.pi)))*math.exp((-(k-m)**2/(2*d*d)))
                    find.append(d)
                elif d==0:
                    if k<=127:
                        p0+=1
                    else:
                        p1+=1
                    continue

                if k<=127:
                    zero+=p

                else:
                    one+=p
            if find:
                max_p=max(find)
                zero+=max_p*p0
                one+=max_p*p1
            if zero>=one:
                part.append(0)
            else:
                part.append(1)
        predict_graph.append(part)

    for i in range(10):
        print(str(i)+':')
        n=0
        for j in predict_graph[i]:

            n+=1
            print(str(j), end=' ')
            if n==28:
                n=0
                print("\n")

def main():
    mode=input("mode:")
    train_feature_file="train-images-idx3-ubyte.gz"
    train_label_file="train-labels-idx1-ubyte.gz"
    test_feature_file="t10k-images-idx3-ubyte.gz"
    test_label_file="t10k-labels-idx1-ubyte.gz"
    imgs, labels, test_imgs, test_labels=read_files(train_feature_file,train_label_file,test_feature_file,test_label_file)

    prior=compute_prior(labels)
    #discrete mode
    if mode == '0':
        bin_imgs=trans_bin(imgs)
        bin_test_imgs=trans_bin(test_imgs)

        likelihood=compute_likelihood(labels,bin_imgs,32)
        predict(bin_test_imgs,test_labels,prior,likelihood)
        graph(likelihood)

    if mode == '1':
        continuous_predict(imgs, labels, test_imgs, test_labels,prior)


if __name__ == "__main__":
    main()