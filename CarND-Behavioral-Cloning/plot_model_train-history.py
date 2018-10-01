#######################################################################################################################
# UDACITY -- CarND--P1--PRJ03_BehavioralCloning -- Additional File For Read & Plot of Model Train History Object File #
# Author : Muthukumar Natarajan                                                                                       #
# Date   : 2018-09-25                                                                                                 #
#######################################################################################################################

### IMPORT Necessary LIBRARIES --> ##########################################################
### Python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

### <-- IMPORT Necessary LIBRARIES ##########################################################


### READ & PLOT MODEL TRAINING HISTORY DATA --> #############################################
### Read Model Training History File
with open('./LeNet5_Train_History.obj', mode='rb') as history_file:
# with open('./NVIDIA_Train_History.obj', mode='rb') as history_file:
	history = pickle.load(history_file) # To Read What Was Written With pickle.dump(history_object.history, history_file)

print()
print("MODEL HISTORY Loaded From *_Train_History.obj")
print()

######### MODEL OUTPUT ['history_object'] VISUALIZATION #########
### Print the Keys contained in the 'history_object.history'
print()
print("Keys Contained in Model Training history_object.history :")
print(history.keys())
print()

### Plot the Training Loss & Validation Loss Vs. Epochs
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Training MSE (Mean Squared Error) Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['Training Dataset', 'Validation Dataset'], loc='upper right')
plt.savefig('LeNet5_Train_Loss_MSE.png')
# plt.savefig('NVIDIA_Train_Loss_MSE.png')
plt.show()
### <-- READ & PLOT MODEL TRAINING HISTORY DATA #############################################
