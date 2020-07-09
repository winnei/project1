from base import Regressor
from data_utils import load_data

# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import LinearSVR

import cv2

import numpy as np
import matplotlib.pyplot as plt

#Display data
data_reg = load_data('./data/regression_data.pkl') #length 4 dictionary: obs, actions, info, dones
data_posbc = load_data('./data/bc_with_gtpos_data.pkl') #length 2 dictionary
data_imgbc = load_data('./data/bc_data.pkl') #length 2 dictionary

"""Regression with image input cannot be done with standard regressors 
Standard regressors have inputs of shape (n_samples, n_features) --> dim 2
Data is still expressed as ONE VECTOR
n_features = no of dimensions of the input in the graphical space
but in the math space (As an array), the input is dim 1

images have shape (n_samples, n_pixels, n_pixels, 3)
cannot be expressed as a vector, unless stacked
but stacking removes locality information

solution: 
- generate a new feature vector of shape (n_samples, n_features)
- bag of classifiers?


Use detectors to search for key points -- find a detector
get location of key points
Pixels around the key points form patches
Encode the patches into feature vectors -- find an encoder (stacking is the easiest)
Collect all the patches (as feature vectors) from all the images, run clustering -- clustering algo
Get K clusters, cluster representatives form bag of features
Classify patches in each image into clusters -- classifying algo
Get a new feature vector of histogram of image (n_samples, n_features = no. of clusters, K) 
Run linear regression on feature vector

"""

class PositionRegressor(Regressor):

    """ Implement solution for Part 1 below  """

    def train(self, data):
        #data is a dictionary with 4 key-value pairs
        data_reg = data

        x_train=data_reg['obs'] #INPUT: numpy array, 500 x 64 x 64 x 3, 500 training examples, coloured, 64x64 pixels, 8 bit

        info=data_reg['info'] #OUTPUT: list of length 500, dictionary containing agent_pos, tuple
        y_train=[]
        for i in info:
            y_train.append(i['agent_pos'])
        
        np.shape(x_train)
        np.shape(y_train)

        # model=LinearRegression()
        # model.fit(x_train,y_train)
        # model=MultiOutputRegressor(GradientBoostingRegressor())
        # model.fit(x_train,y_train)
        # model = KNeighborsRegressor()
        # model.fit(x_train, y_train)
        # model = RandomForestRegressor()
        # model.fit(x_train, y_train)
        # model = LinearSVR()
        # wrapper = MultiOutputRegressor(model)
        # wrapper.fit(x_train, y_train)

        #perform regression on image input and agent position

        #use KAZE To 
        # 1. Find keypoints
        # 2. Describe keypoints (scale and orientation invariant)

        img=x_train[40]
        np.shape(img)
       
        detector=cv2.ORB()
        kp = detector.detectAndCompute(img, None)

        np.shape(kp)
        np.size(kp)
        
        np.shape(des)
        type(des)




       

        print("Using dummy solution for PositionRegressor")
        pass

    def predict(self, Xs):
        return np.zeros((Xs.shape[0], 2))
