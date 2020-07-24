# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:00:54 2019

This module attempts to perform anomalous feedstock detection using a binary
classification model. Random forest binary classification technique has been
employed to classify convolutional feature vectors generated by one of the
standard CNNs.

Run Instructions:
=================
    python binaryClassification.py
    
Output Files:
=============
    - 'RF.pickle'

@author: cgudaval
"""

import numpy as np
import os

import matplotlib.pyplot as plt

#%%
# =============================================================================
# Load features and target variables
# =============================================================================
X_train = np.load(os.path.join(os.getcwd(),'RN_Crop/FD_70', 'LSTM_x_Train.npy'))
X_train = X_train[:,11,:]
y_train = np.load(os.path.join(os.getcwd(),'RN_Crop/FD_70', 'LSTM_y_Train.npy'))

X_test = np.load(os.path.join(os.getcwd(),'RN_Crop/FD_70', 'LSTM_x_Test.npy'))
X_test = X_test[:,11,:]
y_test = np.load(os.path.join(os.getcwd(),'RN_Crop/FD_70', 'LSTM_y_Test.npy'))

#%%
# =============================================================================
# Perform binary classification using random forest classifier
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=10000, max_depth=2, random_state=0)

X_train = np.delete(X_train,np.where(y_train==0)[0][1:10000], axis = 0)
y_train = np.delete(y_train,np.where(y_train==0)[0][1:10000], axis = 0)

RF.fit(X_train, y_train)

# Save RF
import pickle
with open('RF.pickle', 'wb') as handle:
    pickle.dump(RF, handle, protocol=pickle.HIGHEST_PROTOCOL)

## Load RF    
#with open(os.path.join(os.getcwd(),'RF.pickle'), 'rb') as pickle_file:
#    RF = pickle.load(pickle_file)


y_pred = RF.predict(X_test)

accuracy = round(RF.score(X_test,y_test), 4)
#%%
# =============================================================================
# Get Confusion Matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
y_true = y_test
CM = confusion_matrix(y_true, y_pred)#, labels = [1,0])
print(CM)

plt.close('all')
from plotCM import plot_confusion_matrix
plot_confusion_matrix(list(y_true.astype(int)), list(y_pred), classes=['Negative', 'Positive'],normalize=True,
                      title='BinaryClassification')
