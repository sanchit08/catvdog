import numpy as np
import os
import cv2
import random



datadir = "/home/sanchit/Documents/Machine Learing/Datasets/PetImages"

cats = ["Dog", "Cat"]
training_data = []

def create_training_data():
	x=[]
	y=[]
	for category in cats:  # search for all images of dogs and cats

		path = os.path.join(datadir,category)  # create path to dogs and cats
		class_num = cats.index(category)   #0=dog 1=cat
		w = os.listdir(path)
		for img in w:  # iterate over each image per dogs and cats
			try:
				img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
				new_array = cv2.resize(img_array, (100, 100))  # resize to normalize data size
				training_data.append([new_array, class_num])  # add this to our training_data
			except Exception as e: 
				pass
	random.shuffle(training_data)   #shuffle the training data
	for features,label in training_data:
		x.append(features)
		y.append(label)

	x = np.array(x).reshape(-1,100,100,1)    #save data as numpy array
	y = np.array(y)
	
	return x,y
