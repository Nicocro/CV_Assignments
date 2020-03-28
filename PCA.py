import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import tensorflow as tf
import os
import requests
from pathlib import Path
from glob import iglob
import cv2

#Build a dataframe of faces flattened out into 1D arrays.  Output is a dataframe with 
# 1 row for each image and 4096 columns (64x64) of the flattened pixels
faces = pd.DataFrame([])
for path in iglob('Dataset_Faces/SubjectA_GR/*.png'):
 img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
 face = pd.Series(img.flatten(),name=path)
 faces = faces.append(face).astype(int)
 
#print(len(faces))
#print(len(faces.columns))
#print(faces.head())

def displayimages(scatola):
#This function is designed to take a dataframe with faces as an argument, 
# and plot the 20 faces in greyscale in one snapshot
    fig =plt.figure(figsize=(9,13))
    columns = 4
    rows  = 5
    ax=[]
    for i in range(len(scatola)) :
        imageshaped = np.reshape(np.array(scatola.iloc[i]), (64,64))
        ax.append(fig.add_subplot(rows, columns,i+1))
        ax[-1].set_title("fig#"+str(i))
        plt.imshow(imageshaped, cmap='gray')
        
    plt.show()

    return

#displayimages(faces)

#returns a pandas series with key value = imgpath e value column (index=0) with the mean of
#the column 
faces_mean = faces.apply(lambda x: x.mean(), axis=0).astype(int)

#returns a pandas dataframe with centered images (original images - the mean)
faces_centered = faces.subtract(faces_mean, axis = 1)



#displayimages(faces_centered)
#from sklearn.decomposition import PCA
#n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
#faces_pca = PCA(n_components=0.8)
#faces_pca.fit(faces)

#fig, axes = plt.subplots(2,10,figsize=(9,3),
# subplot_kw={'xticks':[], 'yticks':[]},
 #gridspec_kw=dict(hspace=0.01, wspace=0.01))
#for i, ax in enumerate(axes.flat):
# ax.imshow(faces_pca.components_[i].reshape(64,64),cmap='gray')



#Show images as they are loaded in greyscale
#def displayFromDataframe (scatola, pixelsX=int, PixelsY=int):
    #for row in scatola.rows:
        

