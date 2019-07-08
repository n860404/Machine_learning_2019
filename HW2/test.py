
# -*- coding: utf-8 -*-

import struct as st
import numpy as np

filename = {'images' : 'train-images.idx3-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
train_imagesfile = open(filename['images'],'rb')
train_imagesfile.seek(0)
magic = st.unpack('>4B',train_imagesfile.read(4))
nImg = st.unpack('>I',train_imagesfile.read(4))[0]-50000 #num of images
nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column
images_array = np.zeros((nImg,nR,nC))
nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))


import matplotlib.pyplot as plt
plt.imshow(images_array[1,:,:])