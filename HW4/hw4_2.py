# -*- coding: utf-8 -*-

'''EM algorithm'''

import struct
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time

def read_files(train_feature_file,train_label_file):
    with gzip.open(train_feature_file) as fp:
        header = struct.unpack(">IIII", fp.read(16))
        _, N, rows, cols = header

        imgs = np.fromstring(fp.read(N*rows*cols), dtype=np.uint8)

        imgs = imgs.reshape((N, rows*cols))


    with gzip.open(train_label_file) as fp:
        header = struct.unpack(">II", fp.read(8))
        _, N = header

        labels = np.fromstring(fp.read(N), dtype=np.uint8)

    return imgs, labels

def trans_bin(imgs):
    imgs = (imgs >= 128).astype(int)
    return imgs

def mylog(m):
    #return np.log1p(m)
    zthreshold = 10**-10
    return np.log(np.where(m > zthreshold, m, zthreshold))
def main():
    np.set_printoptions(threshold=np.inf)
    start = time.time()
    train_feature_file="train-images-idx3-ubyte.gz"
    train_label_file="train-labels-idx1-ubyte.gz"
    imgs, labels=read_files(train_feature_file,train_label_file)
    imgs=trans_bin(imgs)
    N=60000
    imgs=imgs[:N]
    labels=labels[:N]
    pixcel_num=28*28
    digit_number = 10
    # probability of group
    k = np.ones(digit_number) / digit_number
    # probability of pixel in each group
    p = np.random.rand(digit_number, pixcel_num)
    initial_p = p.copy()

    convergence = 10**-4

    current_convergence = 0
    current_iter = 0

    while True:
        current_iter += 1
        k_pre=k
        p_pre=p

        # E step
        pixel1 = np.matmul(imgs, mylog(p.T))
        pixel0 = np.matmul(1 - imgs, mylog(1 - p.T))
        w = pixel1 + pixel0 + mylog(k)
        w = (w.T - np.max(w, axis=1)).T
        w = np.exp(w)
        w = (w.T / np.sum(w, axis=1)).T

        # M step
        w_sum = np.sum(w, axis=0)

        k = w_sum / w.shape[0]
        p = (np.matmul(imgs.T, w) / w_sum).T

        k_delta = (k - k_pre)
        p_delta = (p - p_pre)


        for i in range(digit_number):
            print("class ",i,":")
            img = np.array(p[i]> 0.5).astype(int).reshape((28,28))
            print(img)


        print("No. of Iteration: ",current_iter)
        #print(p)

        if (np.absolute(k_delta) < convergence).all() and (np.absolute(p_delta) < convergence).all():
            print('delta convergence')
            #print(p)
            break
    p = (p > 0.5).astype(int)
    for i in range(digit_number):
        print("class ",i,":")
        img = np.array(p[i]).reshape((28,28))
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(img, cmap='gray')
        print(img)

        '''
        for j in range(pixcel_num):
            print(p[i][j],end=' ')
            if(j%28==27):
                print('\n')
        print('\n')
        '''

    test_result = np.zeros((10,10))
    test_w = Estep(imgs, k, p)
    test_w_max = np.argmax(test_w, axis=1)

    for i in range(test_result.shape[0]):
        for j in range(test_result.shape[1]):
            test_result[i][j] = np.count_nonzero(labels[test_w_max == i] == j)
    digit_label = np.argmax(test_result, axis=0)
    print(digit_label)
    '''
    print(test_result)

    print(np.argmax(test_result, axis=1))
    print(np.argmax(test_result, axis=0))
    print(np.argmax(test_result, axis=1)[np.argmax(test_result, axis=0)] == range(10))
    '''
    '''
    for i in range(10):
        print("\n----------------------------------------------\n")
        print("Confusion Matrix ",i,":")
        tp = np.count_nonzero( imgs[test_w_max[i] == digit_label[i]] == digit_label[i] )
        fp = np.count_nonzero( imgs[test_w_max[i] == digit_label[i]] != digit_label[i] )
        fn = np.count_nonzero( imgs[test_w_max[i] != digit_label[i]] == digit_label[i] )
        tn = np.count_nonzero( imgs[test_w_max[i] != digit_label[i]] != digit_label[i] )
    '''
    for label,digit in enumerate(digit_label):
        tp = np.count_nonzero( labels[test_w_max == digit] == label )
        fp = np.count_nonzero( labels[test_w_max == digit] != label )
        fn = np.count_nonzero( labels[test_w_max != digit] == label )
        tn = np.count_nonzero( labels[test_w_max != digit] != label )
        print('Confusion Matrix:')
        print('\t\tPredict cluster 1\tPredict cluster 2')
        print('Is cluster 1\t\t',tp,'\t\t\t',fp)
        print('Is cluster 2\t\t',fn,'\t\t\t',tn,'\n')
        print('Sensitivity (Successfully predict cluster 1):',tp / (tp + fn))
        print('Specificity (Successfully predict cluster 2):',tn / (tn + fp))


    end = time.time()
    elapsed = end - start
    print ("Time taken: ", elapsed, "seconds.")



if __name__ == "__main__":
    main()