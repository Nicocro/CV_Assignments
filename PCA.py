import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import requests
from pathlib import Path
from glob import iglob
import cv2

faces = pd.DataFrame([])
for path in iglob('Dataset_Faces/SubjectA_GR/*.png'):
 img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
 face = pd.Series(img.flatten(),name=path)
 faces = faces.append(face)
 
print(len(faces))
print(len(faces.columns))

fig, axes = plt.subplots(10,2,figsize=(9,9),
 subplot_kw={'xticks':[], 'yticks':[]},
 gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
 ax.imshow(faces.iloc[i].values.reshape(64,64),cmap='gray')