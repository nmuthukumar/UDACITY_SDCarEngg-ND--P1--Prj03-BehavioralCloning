##############################################################################################
# UDACITY -- CarND--P1--PRJ03_BehavioralCloning -- Behavior Clone Model for Self-Driving Car #
# Author : Muthukumar Natarajan                                                              #
# Date   : 2018-09-25                                                                        #
##############################################################################################

### IMPORT Necessary LIBRARIES --> ##########################################################
### Python
import numpy as np
import sklearn
import cv2
import matplotlib.pyplot as plt
import pickle

### TensorFlow
import tensorflow as tf

### Keras
import keras
from keras.models import Sequential
from keras.layers.core import Lambda, Activation, Flatten, Dense, Dropout
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.backend import tf as keras_tf
### <-- IMPORT Necessary LIBRARIES ##########################################################


### READ & LOAD TRAINING DATA --> ###########################################################
import csv

### Read Drive Log File Lines
drive_log_lines = []
with open('./data/driving_log.csv') as drive_log_csv_file:
	drive_log_data = csv.reader(drive_log_csv_file)
	for line in drive_log_data:
		# print(line)
		drive_log_lines.append(line)
	del(drive_log_lines[0]) # To Remove the 1st Line with Column Headings.
print()
print("Total Number of Drive Log Lines : ", len(drive_log_lines))
print()

### Split Dataset for Model Train & Validation
from sklearn.model_selection import train_test_split
drive_log_lines_train, drive_log_lines_validation = train_test_split(drive_log_lines, test_size=0.2)
len_drive_log_lines_train      = len(drive_log_lines_train)
len_drive_log_lines_validation = len(drive_log_lines_validation)

### SWITCH TO CHOOSE WHETHER TO WORK ON FULL DATASET OR A SUBSET For Better Performance:
SWITCH_DATA_REDUCE = False # False => FULL DATASET; True => REDUCED DATASET.

if (True == SWITCH_DATA_REDUCE):
	reduction_factor = 0.5 # Change REDUCTION FACTOR As Required.
	len_drive_log_lines_train      = int(len_drive_log_lines_train      * reduction_factor)
	len_drive_log_lines_validation = int(len_drive_log_lines_validation * reduction_factor)
	print()
	print("!!! TAKING ONLY A REDUCED DATASET FOR TRAINING/VALIDATION !!!")
	print("!!! FOR BETTER PERFORMANCE:                               !!!")
	print("    Reduction Factor From Original Dataset : ", reduction_factor)
	print("    Total Number of Drive Log Lines Used   : ", (len_drive_log_lines_train + len_drive_log_lines_validation))
	print()

print()
print("Total Number of Drive Log Lines For Training   : ", len_drive_log_lines_train)
print("Total Number of Drive Log Lines For Validation : ", len_drive_log_lines_validation)
print()
len_train_samples      = (len_drive_log_lines_train      * 6) # 3 Images Per Line * 2 Times Due To Data Augmentation.
len_validation_samples = (len_drive_log_lines_validation * 6) # 3 Images Per Line * 2 Times Due To Data Augmentation.
print()
print("Total Number of Samples For Training   : ", len_train_samples)
print("Total Number of Samples For Validation : ", len_validation_samples)
print()

######### DATA GENERATOR FUNCTION #########
def generator_data(drive_log_lines, batch_size=120): # drive_log_lines => Dataset of Selected Lines From Drive Log CSV File.
	num_lines        = len(drive_log_lines) # _OR_ drive_log_lines.shape[0]
	batch_size_lines = int(batch_size / 6)  # As we Output 3 Images Per Line * 2 Times Due To Data Augmentation.

	# REDUCE THE DATASET If Chosen To
	if (True == SWITCH_DATA_REDUCE):
		new_num_lines     = int(num_lines * reduction_factor)
		selection_indices = np.random.choice(num_lines, size=new_num_lines, replace=False)
		print()
		print()
		print()
		print("!!! TAKING ONLY A REDUCED DATASET, RANDOMLY FOR TRAINING EPOCHS & VALIDATION !!!")
		print("FOR THIS RUN, Selection Indices for Reduced Set of Drive Log Lines: ", selection_indices)
		print()
		# drive_log_lines = drive_log_lines[selection_indices]
		# Above Line results in Error:
		# "TypeError: only integer scalar arrays can be converted to a scalar index"
		# So going for Another Approach:
		drive_log_lines_reduced_set = []
		for index in selection_indices:
			drive_log_lines_reduced_set.append(drive_log_lines[index])

		drive_log_lines = drive_log_lines_reduced_set # To Enable Use of Same Variable Name "drive_log_lines" Further.
		num_lines       = len(drive_log_lines)        # Reduced Dataset's Number of Log Lines.

	ssh_keep_alive = 0 # To Keep AWS SSH Session ON For Whole Model Training Time! See Usage Below in Generator Loop...

	### GENERATOR LOOP
	while 1: # Required To Loop Forever so that the Generator Function Co-Runs As Long As Required By model.fit_generator().
		sklearn.utils.shuffle(drive_log_lines)
		for offset in range(0, num_lines, batch_size_lines):
			batch_lines = drive_log_lines[offset:offset+batch_size_lines]

			### Read-In Drive Log Data -> Camera Images & Steering Angle Measurements
			drive_images             = []
			drive_measurements_steer = []
			for line in batch_lines:
				# Read-In Center Camera Image, Left Camera Image & Right Camera Image
				for i in range(3): # Add Images From All 3 Cameras.
					image_source_path = line[i] # i=0 => Center Image, i=1 => Left Image, i=2 => Right Image.
					tokens            = image_source_path.split('/') # For LINUX.
					# tokens            = image_source_path.split('\\') # For WINDOWS.
					image_filename    = tokens[-1]
					image_local_path  = "./data/IMG/"+image_filename
					# print(image_local_path)
					image             = cv2.imread(image_local_path)
					drive_images.append(image)

				# Read-In Steering Measurement (Corresponding To Center Camera View)
				# and Also Add Augmented Values For Left & Right Camera Views
				meas_steer_cam_center   = float(line[3]) # i=3 => Steering Measurement. Type Cast From string To float.
				                                                                          # Steer Meas. @ Center Cam Perspective.
				steer_angle_diff_offset = 0.3 # Offset For Steer Angle Differences Corresponding To Left & Right Camera Perspectives.
				meas_steer_cam_left     = meas_steer_cam_center + steer_angle_diff_offset # Steer Meas. @ Left   Cam Perspective.
				meas_steer_cam_right    = meas_steer_cam_center - steer_angle_diff_offset # Steer Meas. @ Right  Cam Perspective.
				drive_measurements_steer.append(meas_steer_cam_center) # Add Steering Measurements For All 3 Camera Perspectives.
				drive_measurements_steer.append(meas_steer_cam_left)
				drive_measurements_steer.append(meas_steer_cam_right)

			# print()
			# print("    Original Input Dataset: Batch Number of Images                : ", len(drive_images))
			# print("    Original Input Dataset: Batch Number of Steering Measurements : ", len(drive_measurements_steer))
			# print()

			### Augment the Data for a Better Dataset
			augmented_images             = []
			augmented_measurements_steer = []
			for image, measurement in zip(drive_images, drive_measurements_steer):
				# First Add the Image & Measurement As Is
				augmented_images.append(image)
				augmented_measurements_steer.append(measurement)
				# Then Flip the Image & Negate the Measurement (Data Equivalent To Drive On Reverse Track) and Add these Augmented Data
				flipped_image       = cv2.flip(image, 1) # Flip Around y-Axis (Vertical Axis).
				negated_measurement = (measurement * -1.0)
				augmented_images.append(flipped_image)
				augmented_measurements_steer.append(negated_measurement)

			# print()
			# print("    Augmented Input Dataset: Batch Number of Images                : ", len(augmented_images))
			# print("    Augmented Input Dataset: Batch Number of Steering Measurements : ", len(augmented_measurements_steer))
			# print()

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements_steer)
			# print()
			# print("    X_train Batch Shape : ", X_train.shape)
			# print("    y_train Batch Shape : ", y_train.shape)
			# print()
			# print()
			# print()

			### To Avoid AWS SSH Session Time-Out When No Feedback Messages During Validation!!!:
			print()
			print("!!! KEEPING SSH ALIVE... !!!", ssh_keep_alive) # To Keep AWS SSH Session ON For Whole Model Training Time!
			print()
			ssh_keep_alive = ssh_keep_alive+1

			yield sklearn.utils.shuffle(X_train, y_train) # 'Yield' a Shuffled Set.
### END: def generator_data()

### TRAINING DATA GENERATOR
# Call: generator_data(drive_log_lines_train, batch_size=batch_size)
### VALIDATION DATA GENERATOR
# Call: generator_data(drive_log_lines_validation, batch_size=batch_size)
### <-- READ & LOAD TRAINING DATA ###########################################################


#############################################################################################
### MODEL ###################################################################################
#############################################################################################
### MODEL CHOICES:
LENET5           = 1
NVIDIA_E2E_LEARN = 2
### CHOOSE THE ACTUAL MODEL HERE:
SWITCH_MODEL_CHOICE = NVIDIA_E2E_LEARN

### PARAMETERS:
num_classes =   1
batch_size  = 120
epochs      =   3
dropout_drop_prob = 0.5

"""
### IMAGE PRE-PROCESSOR FOR DRIVE PREDICTION [For Use By <drive.py>]:
### -> NOT REQUIRED Due To Use Of LAMBDA LAYER Inside MODEL, which will be Used For Both Train & Predict!
def image_preprocess(crop_top, crop_bottom, crop_left, crop_right, output_shape):
	# 1. Crop Image
	# 2. Scale Image If Output Size is Different
	# 3. Adjust Image Color Space As Required
	return(image_preprocessed)

def ImagePreprocessForModelPredictInput(image): # Expected Input Image Shape: 160x320x3 (HeightRows x WidthColumns x Channels).
	if (LENET5 == SWITCH_MODEL_CHOICE):
		crop_top     = 64
		crop_bottom  = 24
		crop_left    =  0
		crop_right   =  0
		output_shape = (72,320)
	elif (NVIDIA_E2E_LEARN == SWITCH_MODEL_CHOICE):
		crop_top     = 64
		crop_bottom  = 24
		crop_left    =  0
		crop_right   =  0
		output_shape = (66,200)
	PreprocessedImage = image_preprocess(crop_top, crop_bottom, crop_left, crop_right, output_shape)
	return(PreprocessedImage)
"""

######### MODEL ARCHITECTURE - LENET5 #########
if (LENET5 == SWITCH_MODEL_CHOICE):
	### Create a Keras Sequential Model
	model = Sequential()

	### Pre-Process INPUT Images [In GPU On-the-Fly for Better Performance!]
	# Normalize Input Image, Make Zero Mean
	model.add(Lambda(lambda X: (X/255.0 - 0.5), input_shape=(160,320,3), output_shape=(160,320,3)))
	
	# Crop Images For ROI: Input: 160x320, Output: 72x320.
	crop_top    = 64
	crop_bottom = 24
	crop_left   =  0
	crop_right  =  0
	model.add(Cropping2D(cropping=((crop_top, crop_bottom),(crop_left, crop_right))))
	
	# Scale-Down the Image for Better Performance [Re-Size 72x320 -> 72x160, As Image Width is a bit too Long!]
	# image_in_details = Input(shape=(None, None, 3)) # Images of Arbitrary Shape & 3 Channels.
	# image_out_shape  = (72, 160)
	# model.add(Lambda(lambda image: keras_tf.image.resize_images(image, image_out_shape))(image_in_details))
	model.add(Lambda(lambda image: keras_tf.image.resize_images(image, (72, 160)))) # (72, 160) => Image_Out Shape; Rows x Columns.

	### Layer-1: Convolution, Activation & Max-Pooling
	model.add(Conv2D(filters= 6, kernel_size=(5, 5), strides=1, padding='valid', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	### Layer-2: Convolution, Activation & Max-Pooling
	model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	### Layer-3: Flatten, Full-Connection & Activation
	model.add(Flatten())
	model.add(Dense(120, activation='relu'))

	### Layer-4: Full-Connection & Activation
	model.add(Dense( 84, activation='relu'))
	# Dropout - To Reduce Overfitting & Improve Model Accuracy
	model.add(Dropout(dropout_drop_prob))

	### Layer-5: Full-Connection & OUTPUT
	model.add(Dense(num_classes))

	### Compile the Model
	# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
	model.compile(optimizer='adam', loss='mse')

	### Train the Model
	print()
	print(">>>>>>>>> TRAINING THE MODEL... >>>>>>>>>")
	print()
	# history_object = model.fit(X_train, y_train, batch_size, epochs=epochs, validation_split=0.2, shuffle=True, verbose=1)
	history_object = model.fit_generator(generator=generator_data(drive_log_lines_train, batch_size=batch_size),
		steps_per_epoch=len_train_samples,
		epochs=epochs,
		validation_data=generator_data(drive_log_lines_validation, batch_size=batch_size),
		validation_steps=len_validation_samples,
		verbose=1)

	### Save the Model & history_object
	model.save('Model_LeNet5_BehClone.h5')
	with open('LeNet5_Train_History.obj', mode='wb') as history_file:
		pickle.dump(history_object.history, history_file)
	print()
	print("MODEL SAVED!: Model_LeNet5_BehClone.h5")
	print()

	######### MODEL OUTPUT ['history_object'] VISUALIZATION #########
	### Print the Keys contained in the 'history_object'
	print()
	print(history_object.history.keys())
	print()

	### Plot the Training Loss & Validation Loss Vs. Epochs
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('Model Training MSE (Mean Squared Error) Loss')
	plt.ylabel('MSE Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training Dataset', 'Validation Dataset'], loc='upper right')
	plt.savefig('LeNet5_Train_Loss_MSE.png')
	# plt.show()

### END: if (LENET5 == SWITCH_MODEL_CHOICE)

######### MODEL ARCHITECTURE - NVIDIA_E2E_LEARN #########
elif (NVIDIA_E2E_LEARN == SWITCH_MODEL_CHOICE):
	### Create a Keras Sequential Model
	model = Sequential()

	### Pre-Process INPUT Images [In GPU On-the-Fly for Better Performance!]
	# Normalize Input Image, Make Zero Mean
	model.add(Lambda(lambda X: (X/255.0 - 0.5), input_shape=(160,320,3), output_shape=(160,320,3)))
	
	# Crop Images For ROI: Input: 160x320, Output: 72x320.
	crop_top    = 64
	crop_bottom = 24
	crop_left   =  0
	crop_right  =  0
	model.add(Cropping2D(cropping=((crop_top, crop_bottom),(crop_left, crop_right))))
	
	# Scale-Down the Image for Better Performance [Re-Size To the Image Shape of NVIDIA Camera: (66, 200, 3)]
	# image_in_details = Input(shape=(None, None, 3)) # Images of Arbitrary Shape & 3 Channels.
	# image_out_shape  = (66, 200)
	# model.add(Lambda(lambda image: keras_tf.image.resize_images(image, image_out_shape))(image_in_details))
	model.add(Lambda(lambda image: keras_tf.image.resize_images(image, (66, 200)))) # (66, 200) => Image_Out Shape; Rows x Columns.

	### Layer-1:
	model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=2, padding='valid', activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	### Layer-2:
	model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=2, padding='valid', activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	### Layer-3:
	model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=2, padding='valid', activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	### Layer-4:
	model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))

	### Layer-5:
	model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))

	### Layer-6:
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))

	### Layer-7:
	model.add(Dense(50, activation='relu'))

	### Layer-8:
	model.add(Dense(10, activation='relu'))
	# Dropout - To Reduce Overfitting & Improve Model Accuracy
	# model.add(Dropout(dropout_drop_prob))

	### Layer-9:
	model.add(Dense(num_classes))

	### Compile the Model
	# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
	model.compile(optimizer='adam', loss='mse')

	### Train the Model
	print()
	print(">>>>>>>>> TRAINING THE MODEL... >>>>>>>>>")
	print()
	# history_object = model.fit(X_train, y_train, batch_size, epochs=epochs, validation_split=0.2, shuffle=True, verbose=1)
	history_object = model.fit_generator(generator=generator_data(drive_log_lines_train, batch_size=batch_size),
		steps_per_epoch=len_train_samples,
		epochs=epochs,
		validation_data=generator_data(drive_log_lines_validation, batch_size=batch_size),
		validation_steps=len_validation_samples,
		verbose=1)

	### Save the Model & history_object
	model.save('Model_NVIDIA_BehClone.h5')
	with open('NVIDIA_Train_History.obj', mode='wb') as history_file:
		pickle.dump(history_object.history, history_file)
	print()
	print("MODEL SAVED!: Model_NVIDIA_BehClone.h5")
	print()

	######### MODEL OUTPUT ['history_object'] VISUALIZATION #########
	### Print the Keys contained in the 'history_object'
	print()
	print(history_object.history.keys())
	print()

	### Plot the Training Loss & Validation Loss Vs. Epochs
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('Model Training MSE (Mean Squared Error) Loss')
	plt.ylabel('MSE Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training Dataset', 'Validation Dataset'], loc='upper right')
	plt.savefig('NVIDIA_Train_Loss_MSE.png')
	# plt.show()

### END: if (NVIDIA_E2E_LEARN == SWITCH_MODEL_CHOICE)

### <-- MODEL ###############################################################################
