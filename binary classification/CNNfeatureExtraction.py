# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:52:35 2019

Description:
============
This module genrates convolutional feature vectors for each input image.
Feature vector is the output of CNN that describes the image. There feature
vectors are fed to classification model in furthur stages.

Run Instructions:
=================
    python CNNfeatureExtraction.py --model ResNet

@author: cgudaval
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
import os
import argparse
import numpy as np
import cv2

#%%
# =============================================================================
# Initializations
# =============================================================================
parser = argparse.ArgumentParser(description='This module extracts convolutional feature vectors of feedstock images.')
parser.add_argument('--model', help='Name of CNN model to extract feature vectors',default='ResNet50V2', required=False, type=str)
parser.add_argument('--path', help='Location of folders containing run folders. Eg: "FY18_LT_baseline_1"', default=os.getcwd(), required=False, type=str)
args = parser.parse_args()

model = args.model #'ResNet50V2'
path_data = args.path #os.getcwd(); #os.path.join("C:/","Users","cgudaval", "Desktop", "FCIC", "FCIC_data")

if not os.path.exists(path_data):
    print('Invalid input path.')
    exit()

runs = ["FY18_LT_baseline_1", "FY18_LT_baseline_2", "FY18_LT_baseline_3", "FY18_LT_baseline_4"]
validModels = ['VGG16', 'ResNet', 'ResNet50V2', 'InceptionResNetV2', 'ResNet101V2', 'Xception', 'DenseNet']
#%%
# =============================================================================
# Load CNN model
# =============================================================================
if model == validModels[0]:#'VGG16':
    from tensorflow.keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input
    featureFolderName = 'v2_VGG_npyData_Crop'
    model = VGG16(weights='imagenet', include_top=False)
    model.summary()
    
elif model == validModels[1] or model == validModels[2]:#'ResNet50V2':
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    from keras.applications.resnet_v2 import preprocess_input
    featureFolderName = 'v2_ResNet_npyData_Crop'
    model = ResNet50V2(weights='imagenet', include_top=False)
    model.summary()

elif model == validModels[3]:#'InceptionResNetV2':
    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.inception_resnet_v2 import preprocess_input
    featureFolderName = 'v2_inc_rn'
    model = InceptionResNetV2(weights='imagenet', include_top=False)
    model.summary()
    
elif model == validModels[4]:#'ResNet101V2':
    from tensorflow.keras.applications.resnet_v2 import ResNet101V2
    from keras.applications.resnet_v2 import preprocess_input
    featureFolderName = 'v2_101_ResNet_npyData'
    model = ResNet101V2(weights='imagenet', include_top=False)
    model.summary()
    
elif model == validModels[5]:#'xception':
    from tensorflow.keras.applications.xception import Xception
    from keras.applications.xception import preprocess_input
    featureFolderName = 'xception_npyData'
    model = Xception(weights='imagenet', include_top=False)
    model.summary()
    
elif model == validModels[6]:#'densenet':
    from tensorflow.keras.applications.densenet import DenseNet121
    from keras.applications.densenet import preprocess_input
    featureFolderName = 'densenet_npyData'
    model = DenseNet121(weights='imagenet', include_top=False)
    model.summary()
else:
    print('***Error: Invalid CNN***')
    print('Select CNNs from the below list:\n', validModels)
    exit()
#%%
# =============================================================================
# Extract corresponding convolutional features and save them
# =============================================================================

for dirname in runs:
    path_photos = os.path.join(path_data,dirname,"WB_photos")
    subdirList = os.listdir(path_photos)
    
    for subDir in subdirList:
        if subDir[0] == '.':
            continue
        path_image = os.path.join(path_photos, subDir)
        if not os.path.exists(os.path.join(path_image,featureFolderName)):
            os.mkdir(os.path.join(path_image,featureFolderName))
            print('Created:',os.path.join(path_image,featureFolderName))
        else:
            print('Writing into:',os.path.join(path_image,featureFolderName))
        for fName in os.listdir(path_image):
            if fName[0:2] == '._':
                continue
            if fName.endswith(".JPG") or fName.endswith(".jpg"):
                img_path = os.path.join(path_image, fName)
                
                #Load the image and crop it
                image_cv2 = cv2.imread(img_path)
                row_min = int(image_cv2.shape[0]*0.0); row_max = int(image_cv2.shape[0]*1.0)
                col_min = int(image_cv2.shape[1]*0.33); col_max = int(image_cv2.shape[1]*0.66)
                image_cv2 = image_cv2[row_min:row_max, col_min:col_max]
                
                #TODO: Instead of just cropping images, we can use semantic segmantion techniques to 
                #discard weigh-belt pixels. This would be the best way of refining input data.
                
                #Read cropped image and preprocess it
                img_data = cv2.resize(image_cv2, (224, 224))
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                
                #Feed pre-processed image to CNN and save the extracted feature
                cnn_feature = model.predict(img_data)
                featurePath = os.path.join(path_image,featureFolderName, fName[0:-3]+'npy')
                np.save(featurePath,cnn_feature)
            else:
                print('Found a non-jpg file/folder:',fName) #Printing this to keep track of un-processed file/folder names
