# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:13:17 2021

@author: kaust
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pdb 
filenames = []
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

from sklearn.decomposition import PCA
 
pca = PCA(2) # we need 2 principal components.
converted_data = pca.fit_transform(imag)
converted_data = converted_data.astype(int)
#converted_data.index( 3625.23930485, -1014.59308894)
#pdb.set_trace() 
print(converted_data.shape)

plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
            cmap = c_map)
plt.colorbar()
#plt.xlabel('PC-1') , plt.ylabel('PC-2')
plt.show()


from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors=7) 
  
knn.fit(converted_data,0) 
  
# Calculate the accuracy of the model 
print(knn.score(converted_data, converted_data)) 

#Cluster 1 > 3500
#20210224122638133.jpg, 20210224122638412.jpg, 20210224122621419.jpg, 20210224122621735.jpg


#Cluster 2 
#20210224122604295.jpg, 20210224122605967.jpg, 20210224122605311.jpg, 20210224122603977.jpg


#Cluster 3 < -500
#20210224122608497.jpg, 20210224122607498.jpg, 20210224122629541.jpg, 20210224122648709.jpg



