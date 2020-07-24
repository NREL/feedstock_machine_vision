# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:21:47 2019

This module loads pre-processed input from previous stage and does time-series
classification using GRU neural network.

Run Instructions:
=================
    python timeSeriesClassification.py --alpha 0.00001
    
Output Files:
=============
    All output files are stored in "data" folder.
    
    - 'best_model.hdf5': Trained LSTM or GRU model which can perform time-
                         series classification.
    - 'history.pickle': History file generated during model.fit() operation.
                        This can be used to visualize model behaviour in terms
                        of accuracy and loss during training process.
    
@author: cgudaval
"""
import os
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
import pickle

#%%
# =============================================================================
# Process Input Arguments
# =============================================================================
parser = argparse.ArgumentParser(
        description='This module loads pre-processed input from previous stage '
                    'and does time-series classification using GRU neural network.'
                    )
parser.add_argument('--alpha', help='Learning rate of GRU or LSTM network',\
                    default=0.00001, required=False, type=float)
args = parser.parse_args()

# =============================================================================
# Hyper Parameters to be tuned
# =============================================================================
learningRate = args.alpha

#%%
# =============================================================================
# Load features and target variables
# =============================================================================
X_train = np.load(os.path.join(os.getcwd(), 'data', 'LSTM_x_Train.npy'))
y_train = np.load(os.path.join(os.getcwd(), 'data', 'LSTM_y_Train.npy'))

X_test = np.load(os.path.join(os.getcwd(), 'data', 'LSTM_x_Test.npy'))
y_test = np.load(os.path.join(os.getcwd(), 'data', 'LSTM_y_Test.npy'))

#%%
# =============================================================================
# Build classification model - GRU
# =============================================================================
from tensorflow.keras.layers import GRU, Activation

model = Sequential()
model.add(GRU(50, input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences = True))
model.add(GRU(1, return_sequences = False))
model.add(Activation('sigmoid'))
model.summary()

adam = Adam(lr = learningRate)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
chk = ModelCheckpoint(os.path.join(os.getcwd(),'data','best_model.hdf5'), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
history = model.fit(X_train, y_train, epochs=150,batch_size=128, callbacks=[chk], validation_data=(X_test,y_test), verbose=0)

#%%
# =============================================================================
# Build classification model - LSTM
# =============================================================================
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import Dense
#
#model = Sequential()
#model.add(LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(Dense(1, activation='sigmoid'))
#model.summary()
#
#adam = Adam(lr=0.000001)
#model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
#chk = ModelCheckpoint(os.path.join(os.getcwd(),'best_model_LSTM.hdf5'), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
#history_LSTM = model.fit(X_train, y_train, epochs=100,batch_size=128, callbacks=[chk], validation_data=(X_test,y_test), verbose=0)

#%%
#Save model history
with open(os.path.join(os.getcwd(),'data','history_LSTM.pickle'), 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

##Load model history
#with open(os.path.join(os.getcwd(),'history.pickle'), 'rb') as pickle_file:
#    history = pickle.load(pickle_file)

#%%
#Plot history file
plt.close('all')
x = list(range(1,len(history["accuracy"])+1))
y = history
plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Anomaly detection model")

plt.plot(x, history["accuracy"], label = "training acc")
plt.plot(x, history["val_accuracy"], label = "val_accuracy")

plt.legend()
plt.show()

plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Anomaly detection model")

plt.plot(x, history["loss"], label = "training loss")
plt.plot(x, history["val_loss"], label = "val_loss")

plt.legend()
plt.show()

#%%
# =============================================================================
# Model Testing
# =============================================================================
model = load_model(os.path.join(os.getcwd(),'data','best_model.hdf5'))

from sklearn.metrics import accuracy_score
test_preds = model.predict_classes(X_test)
accuracy_score(y_test, test_preds)

#%%
#Get Confusion Matrix
from sklearn.metrics import confusion_matrix
y_true = y_test
y_pred = test_preds.T[0]
CM = confusion_matrix(y_true, y_pred)#, labels = [1,0])
print("Confusion Matrix:\n",CM)

plt.close('all')
from plotCM import plot_confusion_matrix
plot_confusion_matrix(list(y_true.astype(int)), list(y_pred), classes=['Negative', 'Positive'],normalize=True,
                      title='Confusion Matrix')

#%%
# =============================================================================
# Generalization
# =============================================================================
from AlarmAnalyzer import generalizeAlarm
leftThresh = 2; rightThresh = float("inf");#leftThresh = 2; rightThresh = 12;
result = generalizeAlarm(y_true, y_pred, leftThresh, rightThresh)

CM_Generalised = confusion_matrix(y_true, result)#, labels = [1,0])
print("Generalised Confusion Matrix:\n",CM_Generalised)

plot_confusion_matrix(list(y_true.astype(int)), list(result), classes=['Negative', 'Positive'],normalize=True,
                      title="CM_RN_Gen: left_thresh = {}, right_thresh = {}".format(leftThresh, rightThresh))
        
#%%
#Plot predicted vals and actual vals
fig, axs = plt.subplots(3)
fig.set_size_inches(18,6)
fig.set_tight_layout(True)

axs[0].title.set_text('v2_ResNet_LessLR\nActual Values')
axs[0].fill_between(x = list(range(0,y_test.shape[0])), y1 = y_test, y2 = 0)

axs[1].title.set_text('Predicted Values')
axs[1].fill_between(x = list(range(0,y_test.shape[0])), y1 = test_preds.T[0], y2 = 0)

axs[2].title.set_text('Generalized Predicted Values')
axs[2].fill_between(x = list(range(0,y_test.shape[0])), y1 = result, y2 = 0)
