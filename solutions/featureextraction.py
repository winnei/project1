from base import Regressor
from data_utils import load_data

import cv2

import numpy as np
import matplotlib.pyplot as plt

#Display data
data_reg = load_data('./data/regression_data.pkl') #length 4 dictionary: obs, actions, info, dones
data_posbc = load_data('./data/bc_with_gtpos_data.pkl') #length 2 dictionary
data_imgbc = load_data('./data/bc_data.pkl') #length 2 dictionary

x_train=data_reg['obs'] #INPUT: numpy array, 500 x 64 x 64 x 3, 500 training examples, coloured, 64x64 pixels, 8 bit

info=data_reg['info'] #OUTPUT: list of length 500, dictionary containing agent_pos, tuple
y_train=[]
for i in info:
    y_train.append(i['agent_pos'])
        
img=x_train[40]
print(np.shape(img))


fast = cv2.FastFeatureDetector()
kp = fast.detect(img,None)
print(kp)

