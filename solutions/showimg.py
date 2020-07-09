from base import Regressor
from data_utils import load_data
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_reg = load_data('./data/regression_data.pkl') #length 4 dictionary: obs, actions, info, dones
x_train=data_reg['obs'] #INPUT: numpy array, 500 x 64 x 64 x 3, 500 training examples, coloured, 64x64 pixels, 8 bit
image=x_train[0]
np.shape(image)

# plt.imshow(image) 
# plt.show()

cv2.imshow('image',image)