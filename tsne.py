# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:27:32 2021

@author: kaust
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, discriminant_analysis
import cv2
import os 
import pdb

filenames=[]
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename),0)
            #img = cv2.resize(img, (20, 60)) 
            if img is not None:
                images.append(img)
                filenames.append(filename)
                
    return np.array(images)

image = load_images_from_folder('C:\\Users\\kaust\\Desktop\\Prep for MS\\AWL\\Dimensionality reduction\\morning\\0')
print(image.shape)
imag = image.reshape(len(image), -1)
image_sample = imag[0,:].reshape(50,80)
print(image_sample.shape)

converted_data = manifold.TSNE(n_components=2, init='pca').fit_transform(imag)
converted_data = converted_data.astype(int)
pdb.set_trace() 
print(converted_data.shape)

plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
            cmap = c_map)
plt.colorbar()
#plt.xlabel('PC-1') , plt.ylabel('PC-2')
plt.show()


#Cluster < -700
#20210224122658401.jpg, 20210224122657286.jpg, 20210224122616509.jpg, 20210224122644589.jpg

#Cluster > 3500
#20210224122637118.jpg, 20210224122623801.jpg, 20210224122636669.jpg, 20210224122621952.jpg

#2800 < Cluster < 3100  
#20210224122603559.jpg, 20210224122604893.jpg, 20210224122604792.jpg, 20210224122605410.jpg
